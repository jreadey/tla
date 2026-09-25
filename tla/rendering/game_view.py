"""Phase 2: board + starting fleets rendered. Camera pans; hex size is fixed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import arcade

from tla.ai.policy import NaivePolicy
from tla.battle import RoundResult, apply_battle_outcome, resolve_round
from tla.elevation import marching_squares_segments
from tla.mapgen import filter_islet_contours
from tla.fow import is_hidden, visible_hexes_for
from tla.game_state import BattleLogEntry, GameState, TurnPhase
from tla.hexgrid import AxialCoord, axial_to_pixel, pixel_to_axial
from tla.movement import (
    begin_engagement,
    move_ship_along_path,
    reachable_hexes,
    toggle_submarine_state,
    validate_path,
)
from tla.replay import ReplayWriter
from tla.rendering.hex_render import (
    PATH_HIGHLIGHT_COLOR,
    PATH_LINE_COLOR,
    PLAYER_COLORS,
    RANGE_PREVIEW_COLOR,
    board_pixel_bounds,
    draw_board,
    draw_contour,
    draw_hex_highlight,
    draw_ships,
    draw_sunk_crossbar,
)
from tla.rendering.ship_glyphs import draw_ship_glyph
from tla.ship import Ship, ShipKind
from tla.tile import PLAYER_A, PLAYER_B, PlayerId
from tla.turn_manager import TurnManager

PORT_PANEL_BG_COLOR = (25, 25, 25, 235)
PORT_PANEL_ICON_BOX = 84.0
PORT_PANEL_ICON_HEX_SIZE = 32.0
PORT_PANEL_MARGIN = 14.0
PORT_PANEL_ICON_ROW_HEIGHT = 92.0
PORT_PANEL_HEADER_HEIGHT = 44.0
# The panel is always exactly one glyph wide (a port's build-in-progress is
# a single, automatic, non-interactive value -- see tla.production) --
# this just keeps the title ("Port Production") from feeling cramped.
PORT_PANEL_MIN_WIDTH = 230.0


@dataclass
class ActiveBattle:
    """A battle awaiting the human attacker's stay/retreat decision after
    the round that just happened -- driven by key presses (Enter/Escape)
    rather than a synchronous decision_fn, since the player needs to see
    each round's outcome before choosing."""

    attacker: Ship
    defender: Ship
    rounds: list[RoundResult] = field(default_factory=list)


@dataclass
class Toast:
    """A brief, non-blocking notification (a ship spotted, or first coming
    under attack -- see `_update_spotted_ships`/`_advance_ai_turn`) --
    unlike the sunk/battle/turn-report overlays, this never pauses input;
    it just fades on its own after TOAST_SECONDS."""

    text: str
    remaining: float = 0.0


@dataclass
class SunkMark:
    """One hex's crossbar (see `_draw_sunk_crossbars`) -- who sank there
    (`owners`, for `_sunk_crossbar_color`) and how visible it still is.
    `opacity` starts at 1.0 the turn it's created and fades by
    `SUNK_MARK_FADE_PER_TURN` each subsequent turn boundary (see
    `_dismiss_turn_report`) until it's removed entirely, rather than
    staying at full visibility for one turn and then vanishing outright
    -- user's own request, since a still-solid crossbar can make it hard
    to tell what ship (if any) is actually sitting on that hex now."""

    owners: set[PlayerId]
    opacity: float = 1.0


# Screen pixels/second for keyboard panning (divided by zoom so it always
# feels like the same on-screen speed, not the same world-space speed).
PAN_SPEED = 600.0

_PAN_KEYS = {
    arcade.key.LEFT: (-1, 0),
    arcade.key.A: (-1, 0),
    arcade.key.RIGHT: (1, 0),
    arcade.key.D: (1, 0),
    arcade.key.UP: (0, 1),
    arcade.key.W: (0, 1),
    arcade.key.DOWN: (0, -1),
    arcade.key.S: (0, -1),
}

# zoom > 1 magnifies (zoomed in); zoom < 1 shrinks (zoomed out).
ZOOM_STEP = 1.1
MIN_ZOOM = 0.4
MAX_ZOOM = 3.0

TOOLTIP_BG_COLOR = (25, 25, 25, 230)
TOOLTIP_TEXT_COLOR = arcade.color.WHITE
TOOLTIP_PADDING = 10
TOOLTIP_LINE_HEIGHT = 18
TOOLTIP_WIDTH = 170
TOOLTIP_OFFSET = 16
# Title, Movement, HP, Damage, and one of (ASW | Surfaced/Submerged).
TOOLTIP_MAX_LINES = 5

SUNK_BG_COLOR = (40, 10, 10, 235)
SUNK_BORDER_COLOR = (220, 60, 60)
SUNK_TEXT_COLOR = arcade.color.WHITE

GAME_OVER_BG_COLOR = (10, 10, 10, 245)

SUB_CONTACT_BG_COLOR = (40, 32, 5, 235)
SUB_CONTACT_BORDER_COLOR = (230, 180, 40)
SUB_CONTACT_TEXT_COLOR = arcade.color.WHITE

ATTACK_HIGHLIGHT_COLOR = (255, 40, 40, 130)
ATTACK_BG_COLOR = (40, 10, 10, 235)
ATTACK_BORDER_COLOR = (220, 60, 60)
ATTACK_TEXT_COLOR = arcade.color.WHITE

# Black (not either PLAYER_COLORS entry) marks a hex where both sides lost a
# ship this turn -- see _sunk_crossbar_color.
MUTUAL_SUNK_CROSSBAR_COLOR = (20, 20, 20)
# How much a SunkMark's opacity drops each turn boundary -- see
# _dismiss_turn_report. 0.2 fades a mark out over 5 turns (full visibility
# the turn it's created, then 80%/60%/40%/20% before it's gone).
SUNK_MARK_FADE_PER_TURN = 0.2

TOAST_SECONDS = 4.0
TOAST_BG_COLOR = (20, 20, 20, 220)
TOAST_BORDER_COLOR = (200, 170, 60)
TOAST_WIDTH = 260.0
TOAST_HEIGHT = 26.0
TOAST_MARGIN = 10.0

TURN_REPORT_BG_COLOR = (18, 18, 18, 245)
TURN_REPORT_BORDER_COLOR = (200, 170, 60)
TURN_REPORT_WIDTH = 620.0
TURN_REPORT_PADDING = 20.0
TURN_REPORT_TITLE_HEIGHT = 34.0
TURN_REPORT_STAT_LINE_HEIGHT = 26.0
TURN_REPORT_GLYPH_ROW_HEIGHT = 84.0
TURN_REPORT_FOOTER_HEIGHT = 30.0
TURN_REPORT_GLYPH_HEX_SIZE = 34.0
TURN_REPORT_GLYPH_SPACING = 66.0
TURN_REPORT_SUNK_X_COLOR = (230, 40, 40)
GAME_OVER_TEXT_COLOR = arcade.color.WHITE


class GameView(arcade.View):
    def __init__(
        self,
        game_state: GameState,
        hex_size: float | None = None,
        *,
        replay_path: str | None = None,
        enemy_belief_path: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.game_state = game_state
        board = game_state.board
        self.hex_size = hex_size if hex_size is not None else board.hex_pixel_size

        self.contour_segments = (
            filter_islet_contours(marching_squares_segments(board.elevation), board)
            if board.elevation
            else []
        )

        min_x, min_y, max_x, max_y = board_pixel_bounds(board, self.hex_size)
        self._board_pixel_bounds = (min_x, min_y, max_x, max_y)
        self.camera = arcade.Camera2D(position=((min_x + max_x) / 2, (min_y + max_y) / 2))
        # Screen-space camera for the hover tooltip -- fixed 1:1 with window
        # pixels regardless of the world camera's pan/zoom.
        self.ui_camera = arcade.Camera2D()

        self.turn_manager = TurnManager(game_state)
        # AI opponent: one NaivePolicy instance covers whichever seat(s)
        # game_state.config.player_kinds marks "ai" (see main.py's --ai
        # flag) -- it must be the same instance for the whole game, since
        # it remembers each player's task forces and enemy-fleet belief
        # across turns (see tla.ai.policy.NaivePolicy's own docstring).
        # _ai_turn_iter is the in-progress plan_movement generator being
        # drained one ship per AiConfig.turn_pacing_seconds by on_update,
        # so a human opponent can watch the AI's turn unfold rather than
        # it resolving instantly; None means no AI turn is currently
        # running. enemy_belief_path is None unless --belief was passed,
        # in which case the AI's position-belief field for each tracked
        # enemy ship is dumped to HDF5 as it diffuses (see
        # tla.ai.belief_store) -- closed alongside the replay writer at
        # game end, in on_draw below.
        self.ai_policy = NaivePolicy(enemy_belief_path=enemy_belief_path)
        self._ai_turn_iter: Iterator[None] | None = None
        self._ai_pace_timer: float = 0.0
        # Post-game replay logging (see tla.replay) -- None unless
        # --replay was passed, in which case every half-turn boundary and
        # the game's end get appended to it (see _end_movement_phase and
        # on_draw).
        self.replay_writer: ReplayWriter | None = None
        if replay_path is not None:
            self.replay_writer = ReplayWriter(replay_path)
            # Saved *relative to replay_path's own directory*, not as
            # given on the command line -- replay_gui.py resolves it the
            # same way at read time (against wherever the .jsonl actually
            # is then, not the original cwd), so the pair keeps resolving
            # correctly even if both files are later moved/archived
            # together. Saving the raw --belief string instead would
            # double up the directory when both flags share a common
            # parent (e.g. --replay logs/g.jsonl --belief logs/g.h5 ->
            # naively resolving "logs/g.h5" against "logs/" gives
            # "logs/logs/g.h5"). See ReplayWriter.write_initial's own
            # docstring.
            saved_belief_path = None
            if enemy_belief_path is not None:
                saved_belief_path = os.path.relpath(enemy_belief_path, start=Path(replay_path).parent)
            self.replay_writer.write_initial(game_state, seed=seed, belief_path=saved_belief_path)
        # An in-progress drag: the ship being moved, the exact route drawn
        # so far (starting with its current hex), and a background "how far
        # could I go" hint computed once at drag-start.
        self.drag_ship: Ship | None = None
        self.drag_path: list[AxialCoord] = []
        self.range_preview: dict[AxialCoord, int] = {}
        self.active_battle: ActiveBattle | None = None
        # True the instant contact is made with a previously-hidden
        # submerged submarine, before the first round of that battle has
        # actually been resolved -- see _start_battle. Blocks all input
        # except the dismiss that lets the battle proceed, so the player
        # gets a moment to register what's happening before any damage is
        # dealt, rather than seeing the outcome and the reveal at once.
        self.pending_sub_contact: bool = False
        # Set when a battle just concluded with a sink, dismissed by any
        # key press or click.
        self.sunk_message: str | None = None
        # True once the second mover's (MOVE_B) movement phase is over and
        # the after-action report covering the whole turn is waiting to be
        # dismissed -- see _end_movement_phase. The actual phase transition
        # (production + turn_stats reset) is deferred until dismissal.
        self.pending_turn_report: bool = False
        # Enemy ship ids each player has ever had in fog-of-war vision --
        # used to fire a one-time "<ship> spotted!" toast the moment a new
        # one is first seen. Keyed by viewer, since fog of war (and so
        # what's "new") can differ per player. See _update_spotted_ships.
        self._known_enemy_ship_ids: dict[PlayerId, set[int]] = {}
        self._toasts: list[Toast] = []
        # Ship ids already toasted as "under attack" during the current
        # half-turn's AI draining (see _advance_ai_turn) -- so a multi-round
        # battle only announces itself once, not every round. Reset in
        # _maybe_start_ai_turn alongside _battle_log_watermark.
        self._attack_toasted_ship_ids: set[int] = set()
        # How many of game_state.battle_log's (half-turn-scoped, see
        # GameState.battle_log) entries have already been considered for an
        # attack toast -- battle_log only grows during one player's whole
        # move, so this is the position to resume scanning from on the next
        # _advance_ai_turn step rather than re-scanning entries already
        # handled.
        self._battle_log_watermark: int = 0
        # Set the first time, during the AI's current half-turn draining,
        # that one of the *watching human's own* ships (see
        # _display_player) comes under attack -- the hex to highlight, and
        # a "<attacker> attacks <your ship>!" message. Dismissed by any key
        # press or click, like sunk_message/pending_sub_contact -- without
        # this, a battle against one of your own ships could otherwise
        # flash by during a paced AI turn along with everything else,
        # leaving it unclear which of your ships even got hit. See
        # _advance_ai_turn/_first_attack_on_display_player.
        self.pending_attack_hex: AxialCoord | None = None
        self.pending_attack_message: str | None = None
        # Ship ids already paused-and-highlighted for coming under attack
        # during the current half-turn's AI draining -- companion to
        # _attack_toasted_ship_ids (a separate set: the toast fires for
        # every battle, this only for one involving the display player's
        # own ship), so a multi-round battle only pauses once. Reset in
        # _maybe_start_ai_turn alongside _attack_toasted_ship_ids.
        self._attack_paused_ship_ids: set[int] = set()
        # Hexes where a ship was sunk, accumulated across both halves of
        # the turn *in progress* (see _end_movement_phase, which reads
        # game_state.battle_log -- half-turn-scoped -- into this before
        # TurnManager.end_movement_phase clears it) -- mirrors TurnStats's
        # own whole-turn accumulation window. Keyed by hex, valued by the
        # owner(s) of whichever ship(s) sank there, so a hex where both
        # sides lost a ship (not necessarily to each other, just the same
        # hex sometime this turn) reads as a mutual loss -- see
        # _sunk_crossbar_color.
        self._current_turn_sunk: dict[AxialCoord, set[PlayerId]] = {}
        # Merged in from _current_turn_sunk at the actual turn boundary
        # (see _dismiss_turn_report) as fresh, full-opacity SunkMarks --
        # what's actually drawn on the board (see on_draw/_draw_sunk_
        # crossbars) so the player can see at a glance where losses
        # happened without having to reopen the after-action report.
        # Every existing mark also fades a little at that same boundary
        # (SUNK_MARK_FADE_PER_TURN) and is dropped once fully faded,
        # rather than the older single-turn snapshot this used to be.
        self._sunk_marks: dict[AxialCoord, SunkMark] = {}
        # The empty friendly port currently showing its (read-only) production
        # panel, if any -- opened by clicking it, closed by clicking
        # elsewhere, Escape, or ending movement. There is nothing to choose
        # here: every port automatically builds from the same fixed
        # `ProductionConfig.build_order` (see tla.production) -- this just
        # shows what it's currently building and its banked points.
        self.selected_port: AxialCoord | None = None

        self._held_pan_keys: set[int] = set()
        self._dragging = False
        self._mouse_screen_pos = (0.0, 0.0)
        self._hovered_ship: Ship | None = None

        # arcade.Text objects are reused and repositioned every frame rather
        # than calling arcade.draw_text() fresh each time, which rebuilds a
        # full text layout from scratch and is too slow to do every frame.
        self._hud_texts = [arcade.Text("", 10, 0, arcade.color.WHITE, 13) for _ in range(1)]
        self._tooltip_texts = [
            arcade.Text("", 0, 0, TOOLTIP_TEXT_COLOR, 12) for _ in range(TOOLTIP_MAX_LINES)
        ]
        self._battle_texts = [arcade.Text("", 0, 0, arcade.color.WHITE, 14) for _ in range(4)]
        self._sunk_text = arcade.Text("", 0, 0, SUNK_TEXT_COLOR, 16, anchor_x="center")
        self._game_over_text = arcade.Text(
            "", 0, 0, GAME_OVER_TEXT_COLOR, 36, anchor_x="center", bold=True
        )
        self._port_panel_title_text = arcade.Text("", 0, 0, arcade.color.WHITE, 13)
        self._port_panel_caption_text = arcade.Text(
            "", 0, 0, arcade.color.WHITE, 12, anchor_x="center"
        )
        self._sub_contact_text = arcade.Text(
            "", 0, 0, SUB_CONTACT_TEXT_COLOR, 16, anchor_x="center"
        )
        self._attack_pause_text = arcade.Text("", 0, 0, ATTACK_TEXT_COLOR, 16, anchor_x="center")
        # Reused for up to this many simultaneously-visible toasts (spotted
        # or under-attack); any beyond that just don't get a slot until an
        # older one expires.
        self._toast_texts = [
            arcade.Text("", 0, 0, arcade.color.WHITE, 12) for _ in range(5)
        ]
        self._turn_report_title_text = arcade.Text(
            "", 0, 0, arcade.color.WHITE, 18, anchor_x="center", bold=True
        )
        self._turn_report_player_texts = [
            arcade.Text("", 0, 0, arcade.color.WHITE, 14) for _ in range(2)
        ]
        self._turn_report_none_texts = [
            arcade.Text("(no ships lost)", 0, 0, (170, 170, 170), 12) for _ in range(2)
        ]
        self._turn_report_footer_text = arcade.Text(
            "(press any key to continue)", 0, 0, (190, 190, 190), 12, anchor_x="center"
        )

    def on_show_view(self) -> None:
        self.window.background_color = arcade.color.BLACK
        self._maybe_start_ai_turn()

    def _maybe_start_ai_turn(self) -> None:
        """If it's now an AI-controlled seat's movement phase, start
        draining its movement turn (see `_advance_ai_turn`, called from
        `on_update`). No-op if the current player is human-controlled or
        the game's already over. Production needs no AI decision at all --
        every port, either side, automatically builds from the same fixed
        `ProductionConfig.build_order` (see `tla.production`)."""
        gs = self.game_state
        if gs.winner is not None:
            return
        if gs.config.player_kinds.get(gs.current_player) != "ai":
            return
        self._ai_turn_iter = self.ai_policy.plan_movement(gs, gs.current_player)
        self._ai_pace_timer = 0.0
        # gs.battle_log was just cleared for this half-turn (see
        # tla.turn_manager.TurnManager.end_movement_phase) -- start
        # scanning it from the beginning, with a clean slate of which ships
        # have already gotten an under-attack toast/pause this half-turn.
        self._battle_log_watermark = 0
        self._attack_toasted_ship_ids = set()
        self._attack_paused_ship_ids = set()

    def _advance_ai_turn(self, delta_time: float) -> None:
        """Drain one step of `_ai_turn_iter` every `AiConfig.turn_pacing_
        seconds`, so a human opponent can watch the AI's turn unfold ship
        by ship. Ends the AI's movement phase itself once the generator is
        exhausted -- mirroring exactly what the Enter key does for a human
        -- and immediately checks for another AI seat, covering an
        AI-vs-AI handoff.

        Each step also fires a one-time, non-blocking "<ship> attacks
        <ship>!" toast for any battle that started this step (see
        `_toast_new_attacks`) -- otherwise it's hard to follow which of the
        AI's ships are even fighting during a paced turn you're only half
        watching.

        If that step sank a ship, draining pauses there and shows the same
        blocking sunk-ship overlay a human's own battles use (see
        `on_key_press`/`on_mouse_press`) -- otherwise a ship lost while it
        wasn't your move (an AI attack, or a scout it sent ahead) would
        just flash by unnoticed during a paced turn you're only half
        watching. Paused for as long as `sunk_message` is set; resumes
        automatically once the human dismisses it, since this is called
        again every `on_update` tick regardless.

        Failing that, if this step is the first attack against one of the
        *watching human's own* ships, draining pauses and highlights that
        hex instead (see `_first_attack_on_display_player`) -- same
        reasoning as the sunk pause: during a paced AI turn you're only
        half watching, it's otherwise easy to miss which of your ships
        just came under fire until the outcome (or a sinking) is already
        old news."""
        if self.sunk_message is not None or self.pending_attack_hex is not None:
            return
        self._ai_pace_timer += delta_time
        pacing = self.game_state.config.ai.turn_pacing_seconds
        while self._ai_turn_iter is not None and self._ai_pace_timer >= pacing:
            self._ai_pace_timer -= pacing
            before = {sid: (s.owner, s.kind) for sid, s in self.game_state.ships.items()}
            try:
                next(self._ai_turn_iter)
            except StopIteration:
                self._ai_turn_iter = None
                self._end_movement_phase()
                return
            entries = self.game_state.battle_log[self._battle_log_watermark :]
            self._battle_log_watermark = len(self.game_state.battle_log)
            self._toast_new_attacks(entries)
            sunk = [owner_kind for sid, owner_kind in before.items() if sid not in self.game_state.ships]
            if sunk:
                self.sunk_message = self._sunk_message_for(sunk)
                return
            attack = self._first_attack_on_display_player(entries)
            if attack is not None:
                self.pending_attack_hex, self.pending_attack_message = attack
                return

    def _toast_new_attacks(self, entries: list[BattleLogEntry]) -> None:
        """Fire a one-time, non-blocking "<ship> attacks <ship>!" toast the
        first time each ship enters combat during the AI's current
        half-turn (as attacker or defender) -- a later round of the same,
        still-ongoing battle doesn't re-announce either ship (only skipped
        once BOTH ships in a round have already been toasted, so a ship
        that gets attacked again later by a *different* ship still gets a
        fresh toast). `entries` is the slice of `game_state.battle_log`
        (see `tla.battle.resolve_round`) new since the last check -- see
        `_advance_ai_turn`, the sole caller, which owns the watermark this
        is read from."""
        for entry in entries:
            if entry.attacker_id in self._attack_toasted_ship_ids and entry.defender_id in self._attack_toasted_ship_ids:
                continue
            self._attack_toasted_ship_ids.add(entry.attacker_id)
            self._attack_toasted_ship_ids.add(entry.defender_id)
            attacker_label = self._label_for(entry.attacker_owner, entry.attacker_kind)
            defender_label = self._label_for(entry.defender_owner, entry.defender_kind)
            self._toasts.append(Toast(text=f"{attacker_label} attacks {defender_label}!", remaining=TOAST_SECONDS))

    def _first_attack_on_display_player(self, entries: list[BattleLogEntry]) -> tuple[AxialCoord, str] | None:
        """The (hex, message) for the first `entries` battle where the
        *watching human's own* ship (see `_display_player`) is the
        defender -- None if none qualify. Only the defender side, never
        the attacker: during the AI's own turn only AI ships move, so any
        battle it starts necessarily has an AI ship as attacker -- the
        display player's ship, if involved at all, is always the one
        getting hit. Deduplicated by `_attack_paused_ship_ids` the same
        way `_toast_new_attacks` dedupes toasts, but tracked separately:
        this only ever fires for the display player's own ship, a strict
        subset of what gets toasted, so the two sets naturally diverge."""
        display_player = self._display_player()
        for entry in entries:
            if entry.defender_owner != display_player:
                continue
            if entry.defender_id in self._attack_paused_ship_ids:
                continue
            self._attack_paused_ship_ids.add(entry.defender_id)
            attacker_label = self._label_for(entry.attacker_owner, entry.attacker_kind)
            defender_kind_label = entry.defender_kind.value.replace("_", " ").title()
            return entry.battle_hex, f"{attacker_label} attacks your {defender_kind_label}!"
        return None

    def _posture_snapshot(self) -> dict[PlayerId, dict]:
        """Both players' current global-layer posture (see
        `tla.ai.global_strategy`), for `ReplayWriter.write_half_turn`/
        `write_final`'s optional `posture` param -- filters out `None`
        (a human-controlled or not-yet-planned-for player), the same
        omit-if-absent shape `task_forces` already uses there."""
        snapshots = {
            player: self.ai_policy.posture_snapshot_for(player) for player in (PLAYER_A, PLAYER_B)
        }
        return {player: snapshot for player, snapshot in snapshots.items() if snapshot is not None}

    def _end_movement_phase(self) -> None:
        """End the current player's movement phase -- the shared path for
        both the human's Enter key and an AI's turn finishing. If this is
        the *second* mover (MOVE_B) ending, a full turn's worth of battles
        is now complete, so the after-action report is shown first (see
        _dismiss_turn_report); the actual phase transition -- which runs
        production and resets turn_stats -- is deferred until it's
        dismissed, so the report still has the turn's real numbers to
        read. Ending the first mover's (MOVE_A) phase has nothing to
        report yet and proceeds immediately, as before."""
        if self.replay_writer is not None:
            # Recorded here, before either branch below runs, so phase/
            # current_player/turn_stats still reflect the half-turn that
            # just finished -- the MOVE_B branch doesn't mutate game_state
            # itself (it only defers to _dismiss_turn_report), so this is
            # correct and sufficient for both branches.
            self.replay_writer.write_half_turn(
                self.game_state,
                phase=self.game_state.phase,
                player=self.game_state.current_player,
                task_forces=self.ai_policy.task_forces_for(PLAYER_A) + self.ai_policy.task_forces_for(PLAYER_B),
                posture=self._posture_snapshot(),
            )
        # Same "before battle_log is cleared" reasoning as the replay write
        # above -- see _record_turn_sunk_hexes.
        self._record_turn_sunk_hexes()
        if self.game_state.phase == TurnPhase.MOVE_B:
            self.pending_turn_report = True
            return
        self.turn_manager.end_movement_phase()
        self._maybe_start_ai_turn()

    def _record_turn_sunk_hexes(self) -> None:
        """Merge every sinking in `game_state.battle_log` (half-turn-
        scoped -- cleared by the `TurnManager.end_movement_phase` call
        this always runs just before, in `_end_movement_phase`) into
        `_current_turn_sunk`, keyed by hex and valued by the sunk ship's
        owner. Called once per half-turn, so a full turn's worth (both
        movement phases) accumulates before `_dismiss_turn_report` merges
        it into `_sunk_marks` for display -- see that method and
        `_draw_sunk_crossbars`."""
        for entry in self.game_state.battle_log:
            owners: set[PlayerId] = set()
            if entry.attacker_sunk:
                owners.add(entry.attacker_owner)
            if entry.defender_sunk:
                owners.add(entry.defender_owner)
            if not owners:
                continue
            self._current_turn_sunk.setdefault(entry.battle_hex, set()).update(owners)

    def _dismiss_turn_report(self) -> None:
        self.pending_turn_report = False
        # Every existing mark fades a step first -- including ones from
        # the turn just ending, so a mark isn't visible at full strength
        # for two turns in a row -- then what just accumulated in
        # _current_turn_sunk (both halves of the turn now ending) is
        # merged in as fresh, full-opacity marks, overwriting a fading
        # mark at the same hex rather than compounding with it (a new
        # sinking there is a new event, not a continuation of the old
        # one). See _draw_sunk_crossbars for the actual drawing.
        for mark in self._sunk_marks.values():
            mark.opacity -= SUNK_MARK_FADE_PER_TURN
        self._sunk_marks = {h: m for h, m in self._sunk_marks.items() if m.opacity > 0}
        for hex_coord, owners in self._current_turn_sunk.items():
            self._sunk_marks[hex_coord] = SunkMark(owners=owners)
        self._current_turn_sunk = {}
        self.turn_manager.end_movement_phase()
        self._maybe_start_ai_turn()

    def on_resize(self, width: int, height: int) -> None:
        # Keep the camera's current position -- only the viewport/projection
        # need to match the new window size, so panning isn't reset.
        self.camera.match_window(position=False)
        self.ui_camera.match_window(position=True)
        self._grow_zoom_to_fill_window(width, height)

    def _grow_zoom_to_fill_window(self, width: int, height: int) -> None:
        """If the window has grown past the map's natural size at the
        current zoom, zoom in just enough that the map keeps filling the
        frame -- otherwise a bigger window just reveals more blank canvas
        around the same content, and the player has to zoom/pan manually
        to compensate (which is exactly the reported annoyance).

        Never zooms *out* on a resize: a map that needs panning to see in
        full at its native size (see app.py's initial window sizing,
        which deliberately never shrinks hexes to force a big map to fit
        on screen) keeps that behavior unchanged, and a deliberate manual
        zoom-in the player already made is never undone by an incidental
        resize.
        """
        min_x, min_y, max_x, max_y = self._board_pixel_bounds
        natural_width = max_x - min_x
        natural_height = max_y - min_y
        if natural_width <= 0 or natural_height <= 0:
            return
        cover_zoom = max(width / natural_width, height / natural_height)
        self.camera.zoom = max(self.camera.zoom, min(cover_zoom, MAX_ZOOM))

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if self.game_state.winner is not None:
            return
        if self.sunk_message is not None:
            # Checked BEFORE the _ai_turn_iter guard below: a sink during
            # the AI's own turn (see _advance_ai_turn) pauses draining and
            # sets this with _ai_turn_iter still non-None -- if the guard
            # ran first, the human could never dismiss it and the game
            # would be stuck. A human-driven sink (_conclude_battle) only
            # ever happens with _ai_turn_iter already None, so this
            # ordering is safe for both cases.
            self.sunk_message = None
            return
        if self.pending_attack_hex is not None:
            # Same ordering reasoning as sunk_message just above -- set
            # during the AI's own turn with _ai_turn_iter still non-None.
            self.pending_attack_hex = None
            self.pending_attack_message = None
            return
        if self._ai_turn_iter is not None:
            return  # an AI seat's paced turn is playing out -- not the human's input to give
        if self.pending_turn_report:
            self._dismiss_turn_report()
            return
        if self.pending_sub_contact:
            self.pending_sub_contact = False
            self._resolve_battle_round()
            return
        if self.active_battle is not None:
            if symbol == arcade.key.ENTER:
                self._resolve_battle_round()
            elif symbol == arcade.key.ESCAPE:
                self._conclude_battle(retreated=True)
            return

        if symbol in _PAN_KEYS:
            self._held_pan_keys.add(symbol)
        elif symbol == arcade.key.F11:
            self.window.set_fullscreen(not self.window.fullscreen)
        elif symbol == arcade.key.ESCAPE:
            if self.window.fullscreen:
                self.window.set_fullscreen(False)
            elif self.drag_ship is not None:
                self._abort_drag()
            elif self.selected_port is not None:
                self._close_port_panel()
        elif symbol in (arcade.key.PLUS, arcade.key.EQUAL, arcade.key.NUM_ADD):
            self._zoom_toward_screen_point(self.window.width / 2, self.window.height / 2, ZOOM_STEP)
        elif symbol in (arcade.key.MINUS, arcade.key.NUM_SUBTRACT):
            self._zoom_toward_screen_point(self.window.width / 2, self.window.height / 2, 1 / ZOOM_STEP)
        elif symbol == arcade.key.KEY_0:
            self.camera.zoom = 1.0
        elif symbol == arcade.key.ENTER:
            self._abort_drag()
            self._close_port_panel()
            self._end_movement_phase()
        elif symbol == arcade.key.T:
            self._toggle_hovered_submarine()

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self._held_pan_keys.discard(symbol)

    def _toggle_hovered_submarine(self) -> None:
        """T toggles surfaced/submerged for whichever of your own submarines
        is under the cursor -- independent of the drag-to-move gesture,
        since there's no natural pause in a press-drag-release move to
        squeeze a key press into otherwise."""
        ship = self._hovered_ship
        if ship is None or ship.kind != ShipKind.SUBMARINE or ship.owner != self.game_state.current_player:
            return
        stats = self.game_state.config.ship_stats.stats[ship.kind]
        try:
            toggle_submarine_state(ship, stats)
        except ValueError:
            return
        if self.drag_ship is ship:
            self._abort_drag()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if self.game_state.winner is not None:
            return
        if self.sunk_message is not None:
            # See the matching comment in on_key_press for why this must
            # be checked before the _ai_turn_iter guard.
            self.sunk_message = None
            return
        if self.pending_attack_hex is not None:
            self.pending_attack_hex = None
            self.pending_attack_message = None
            return
        if self._ai_turn_iter is not None:
            return  # an AI seat's paced turn is playing out -- not the human's input to give
        if self.pending_turn_report:
            self._dismiss_turn_report()
            return
        if self.pending_sub_contact:
            self.pending_sub_contact = False
            self._resolve_battle_round()
            return
        if button == arcade.MOUSE_BUTTON_RIGHT:
            self._dragging = True
            return
        if button != arcade.MOUSE_BUTTON_LEFT or self.active_battle is not None:
            return
        if self.selected_port is not None:
            # The panel is read-only (nothing to click -- see tla.production),
            # so any click within its bounds is simply consumed rather than
            # falling through to _start_drag, which would reinterpret it as
            # a click on whatever map hex happens to be underneath the
            # panel. Except when that hex holds the player's own movable
            # ship: the panel is drawn at a fixed screen position unrelated
            # to where the selected port actually sits on the map, so it
            # can coincidentally overlap an unrelated ship elsewhere on
            # screen -- a click meant to drag that ship must still win, or
            # the ship becomes stuck with no visible cause.
            if self._point_in_port_panel(x, y) and self._draggable_ship_at_screen_point(x, y) is None:
                return
        self._start_drag(x, y)

    def _draggable_ship_at_screen_point(self, screen_x: float, screen_y: float) -> Ship | None:
        """The current player's own ship at this screen position, if it
        still has movement left this turn -- i.e. exactly what a click
        there would start dragging (see _start_drag). Used by
        on_mouse_press to let such a click win over an open port panel's
        fixed-position bounds check."""
        gs = self.game_state
        world = self.camera.unproject((screen_x, screen_y))
        hex_coord = pixel_to_axial(world[0], world[1], self.hex_size)
        ship = gs.ship_at(hex_coord)
        if ship is not None and ship.owner == gs.current_player and ship.movement_remaining > 0:
            return ship
        return None

    def _open_port_panel(self, coord: AxialCoord) -> None:
        self.selected_port = coord

    def _close_port_panel(self) -> None:
        self.selected_port = None

    def _start_drag(self, screen_x: float, screen_y: float) -> None:
        gs = self.game_state
        world = self.camera.unproject((screen_x, screen_y))
        hex_coord = pixel_to_axial(world[0], world[1], self.hex_size)

        ship = gs.ship_at(hex_coord)
        if ship is not None and ship.owner == gs.current_player and ship.movement_remaining > 0:
            self._close_port_panel()
            self.drag_ship = ship
            self.drag_path = [hex_coord]
            self.range_preview = reachable_hexes(
                ship, gs, treat_as_open=self._hidden_submerged_sub_positions()
            )
            return

        tile = gs.board.get_tile(hex_coord)
        if ship is None and tile is not None and tile.is_port and tile.port_display_owner == gs.current_player:
            if self.selected_port == hex_coord:
                self._close_port_panel()
            else:
                self._open_port_panel(hex_coord)
            return

        self._close_port_panel()

    def _extend_drag(self, screen_x: float, screen_y: float) -> None:
        world = self.camera.unproject((screen_x, screen_y))
        hex_coord = pixel_to_axial(world[0], world[1], self.hex_size)

        if hex_coord == self.drag_path[-1]:
            return
        if hex_coord in self.drag_path:
            # Dragging back over an already-drawn hex undoes the path back
            # to that point, rather than requiring a precise one-step undo.
            index = self.drag_path.index(hex_coord)
            self.drag_path = self.drag_path[: index + 1]
            return
        trial_path = self.drag_path + [hex_coord]
        try:
            # The newest hex is only provisionally the end of the drag --
            # the player may keep going -- so a friendly-occupied hex is
            # tolerated here even though it could never actually be the
            # final stop; _commit_drag's move call enforces that for real.
            # treat_as_open likewise previews a hidden submerged sub's hex
            # as if it were empty, so its exact location can't be inferred
            # from where the drag would otherwise refuse to extend --
            # _commit_drag snaps the committed path back to the real thing.
            validate_path(
                self.drag_ship,
                trial_path,
                self.game_state,
                allow_passthrough_final=True,
                treat_as_open=self._hidden_submerged_sub_positions(),
            )
        except ValueError:
            return
        self.drag_path = trial_path

    def _hidden_submerged_sub_positions(self) -> frozenset[AxialCoord]:
        """Positions of enemy submerged submarines currently hidden by fog
        of war. Used only to make the drag preview (range highlight and
        incremental path validation) behave as if those hexes were empty,
        so their exact location can't be deduced from where movement would
        otherwise be blocked -- the actual committed move is snapped back
        to reality by `_truncate_path_at_first_hidden_sub`. Empty if fog of
        war is disabled, since then nothing is hidden in the first place."""
        gs = self.game_state
        if not gs.config.fow.enabled:
            return frozenset()
        return frozenset(
            s.position
            for s in gs.ships.values()
            if s.owner != gs.current_player and s.kind == ShipKind.SUBMARINE and not s.surfaced
        )

    def _truncate_path_at_first_hidden_sub(self, ship: Ship, path: list[AxialCoord]) -> list[AxialCoord]:
        """If the drawn `path` runs through or ends on a hex holding a
        submerged enemy submarine -- hidden during the drag preview, see
        `_hidden_submerged_sub_positions` -- truncate it at the first one
        encountered: the player couldn't see it coming, so their ship
        makes contact and stops (triggering a battle) there instead of
        sailing straight through to wherever they actually aimed."""
        gs = self.game_state
        if not gs.config.fow.enabled:
            return path
        for i in range(1, len(path)):
            occupant = gs.ship_at(path[i])
            if (
                occupant is not None
                and occupant.owner != ship.owner
                and occupant.kind == ShipKind.SUBMARINE
                and not occupant.surfaced
            ):
                return path[: i + 1]
        return path

    def _commit_drag(self) -> None:
        ship = self.drag_ship
        path = self.drag_path
        self._abort_drag()
        if ship is None or len(path) < 2:
            return
        path = self._truncate_path_at_first_hidden_sub(ship, path)

        if self.game_state.ship_at(path[-1]) is not None:
            try:
                defender = begin_engagement(ship, path, self.game_state)
            except ValueError:
                return
            self._start_battle(ship, defender)
        else:
            try:
                # Captures a port and ends the game on the spot if that
                # completes total port control -- see move_ship_along_path.
                move_ship_along_path(ship, path, self.game_state)
            except ValueError:
                return

    def _abort_drag(self) -> None:
        self.drag_ship = None
        self.drag_path = []
        self.range_preview = {}

    def _start_battle(self, attacker: Ship, defender: Ship) -> None:
        self.active_battle = ActiveBattle(attacker=attacker, defender=defender)
        if defender.kind == ShipKind.SUBMARINE and not defender.surfaced:
            # First contact with a submerged sub: pause for acknowledgment
            # before the first round -- which always happens the instant
            # contact is made (see battle.run_battle) -- deals any damage,
            # so the player has a moment to register what's going on
            # instead of seeing the reveal and the outcome simultaneously.
            # Dismissed by any key/click, see on_key_press/on_mouse_press.
            self.pending_sub_contact = True
            return
        self._resolve_battle_round()

    def _resolve_battle_round(self) -> None:
        battle = self.active_battle
        round_result = resolve_round(battle.attacker, battle.defender, self.game_state)
        battle.rounds.append(round_result)
        if round_result.attacker_sunk or round_result.defender_sunk:
            self._conclude_battle(retreated=False)

    def _conclude_battle(self, retreated: bool) -> None:
        battle = self.active_battle
        attacker, defender = battle.attacker, battle.defender
        self.sunk_message = self._sunk_message(attacker, defender)
        if defender.is_sunk and self._hovered_ship is defender:
            self._hovered_ship = None
        if attacker.is_sunk and self._hovered_ship is attacker:
            self._hovered_ship = None
        # Removes sunk ship(s), repositions a surviving attacker (which may
        # capture a port), and refreshes game_state.winner -- see
        # tla.battle.apply_battle_outcome. A pure retreat is a no-op here.
        apply_battle_outcome(self.game_state, attacker, defender)
        self.active_battle = None

    def _sunk_message(self, attacker: Ship, defender: Ship) -> str | None:
        if attacker.is_sunk and defender.is_sunk:
            return f"{self._ship_label(attacker)} and {self._ship_label(defender)} both sunk!"
        if attacker.is_sunk:
            return f"{self._ship_label(attacker)} sunk!"
        if defender.is_sunk:
            return f"{self._ship_label(defender)} sunk!"
        return None

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == arcade.MOUSE_BUTTON_LEFT:
            self._commit_drag()
        if button == arcade.MOUSE_BUTTON_RIGHT:
            self._dragging = False

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if self._dragging:
            cx, cy = self.camera.position
            zoom = self.camera.zoom
            self.camera.position = (cx - dx / zoom, cy - dy / zoom)
        if self.drag_ship is not None:
            self._extend_drag(x, y)
        self._update_hover(x, y)

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self._update_hover(x, y)

    def _update_hover(self, screen_x: float, screen_y: float) -> None:
        self._mouse_screen_pos = (screen_x, screen_y)
        world = self.camera.unproject((screen_x, screen_y))
        hex_coord = pixel_to_axial(world[0], world[1], self.hex_size)
        gs = self.game_state
        ship = gs.ship_at(hex_coord)
        if ship is not None and ship.owner != self._display_player():
            visible = self._visible_hexes()
            if visible is not None and not self._is_ship_visible(ship, visible):
                ship = None  # hidden by fog of war -- no tooltip, no "T" toggle target
        self._hovered_ship = ship

    def _is_ship_visible(self, ship: Ship, visible_hexes: set[AxialCoord]) -> bool:
        """Whether an enemy `ship` should currently be shown, given fog of
        war is enabled (`visible_hexes` is `_display_player`'s vision). The
        one exception to the normal fog rules (including the
        submerged-submarine override -- see tla.fow.is_hidden): a ship
        currently being fought is always shown, since direct combat
        contact is the only way to spot a submerged sub in the first
        place."""
        if self.active_battle is not None and ship is self.active_battle.defender:
            return True
        return not is_hidden(self._display_player(), ship, visible_hexes)

    def _display_player(self) -> PlayerId:
        """Whose fog-of-war perspective the screen should currently show.
        Normally whoever's turn it is (`current_player`) -- correct for
        two-human hotseat play, where the screen always represents
        whichever player is at the keyboard right now. But while an AI
        seat is playing out its own turn, `current_player` is the AI, and
        showing *its* vision would hand the watching human a free look at
        everything the AI can see -- including ships fog of war would
        otherwise hide from them. Whenever exactly one seat is
        human-controlled, that seat's own vision is used instead,
        regardless of whose turn it technically is."""
        gs = self.game_state
        human_players = [p for p, kind in gs.config.player_kinds.items() if kind == "human"]
        if len(human_players) == 1:
            return human_players[0]
        return gs.current_player

    def _visible_hexes(self) -> set[AxialCoord] | None:
        """`_display_player`'s fog-of-war vision, or None if fog of war is
        disabled -- in which case callers should treat everything as
        visible."""
        gs = self.game_state
        if not gs.config.fow.enabled:
            return None
        return visible_hexes_for(gs, self._display_player())

    def _ship_label(self, ship: Ship) -> str:
        return self._label_for(ship.owner, ship.kind)

    def _label_for(self, owner: PlayerId, kind: ShipKind) -> str:
        owner_label = "Player A" if owner == PLAYER_A else "Player B"
        kind_label = kind.value.replace("_", " ").title()
        return f"{owner_label} {kind_label}"

    def _sunk_message_for(self, sunk: list[tuple[PlayerId, ShipKind]]) -> str:
        """Same "<ship> sunk!" / "<ship> and <ship> both sunk!" phrasing as
        `_sunk_message`, but built from bare (owner, kind) pairs rather
        than live Ship objects -- used for a sink discovered during the
        AI's own turn (see `_advance_ai_turn`), where the ships involved
        are already gone from `game_state.ships` by the time it's noticed."""
        labels = [self._label_for(owner, kind) for owner, kind in sunk]
        if len(labels) == 2:
            return f"{labels[0]} and {labels[1]} both sunk!"
        return f"{labels[0]} sunk!"

    def _update_toasts(self, delta_time: float) -> None:
        """Age every current toast (spotted-ship or under-attack -- see
        `_update_spotted_ships`/`_advance_ai_turn`) and drop any that have
        faded out. Called once per frame regardless of what's producing
        toasts, so timing is smooth and independent of the AI's own
        (slower, paced) update cadence."""
        for toast in self._toasts:
            toast.remaining -= delta_time
        self._toasts = [t for t in self._toasts if t.remaining > 0]

    def _update_spotted_ships(self) -> None:
        """Fire a one-time, non-blocking "<ship> spotted!" toast the
        moment an enemy ship is first seen in the display player's fog of
        war vision (never for a submerged submarine -- fow.is_hidden keeps
        those out of `visible` entirely regardless, so a combat reveal is
        the SUB CONTACT flow's job, not this one). No-op if fog of war is
        disabled -- there's no "first sighting" moment without it."""
        gs = self.game_state
        if not gs.config.fow.enabled:
            return
        display_player = self._display_player()
        visible = visible_hexes_for(gs, display_player)
        known = self._known_enemy_ship_ids.setdefault(display_player, set())
        for ship in gs.ships.values():
            if ship.owner == display_player or ship.id in known:
                continue
            if is_hidden(display_player, ship, visible):
                continue
            known.add(ship.id)
            self._toasts.append(
                Toast(text=f"{self._ship_label(ship)} spotted!", remaining=TOAST_SECONDS)
            )

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        if scroll_y > 0:
            self._zoom_toward_screen_point(x, y, ZOOM_STEP)
        elif scroll_y < 0:
            self._zoom_toward_screen_point(x, y, 1 / ZOOM_STEP)

    def _zoom_toward_screen_point(self, screen_x: float, screen_y: float, factor: float) -> None:
        """Change zoom by `factor`, keeping the world point under
        (screen_x, screen_y) fixed on screen -- so scrolling over a spot on
        the map zooms toward that spot rather than the window's center."""
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.camera.zoom * factor))
        if new_zoom == self.camera.zoom:
            return
        world_before = self.camera.unproject((screen_x, screen_y))
        self.camera.zoom = new_zoom
        world_after = self.camera.unproject((screen_x, screen_y))
        cx, cy = self.camera.position
        self.camera.position = (
            cx + (world_before[0] - world_after[0]),
            cy + (world_before[1] - world_after[1]),
        )

    def on_update(self, delta_time: float) -> None:
        if self._ai_turn_iter is not None:
            self._advance_ai_turn(delta_time)
        self._update_toasts(delta_time)
        self._update_spotted_ships()
        if not self._held_pan_keys:
            return
        move_x = sum(dx for key, (dx, _) in _PAN_KEYS.items() if key in self._held_pan_keys)
        move_y = sum(dy for key, (_, dy) in _PAN_KEYS.items() if key in self._held_pan_keys)
        if move_x == 0 and move_y == 0:
            return
        cx, cy = self.camera.position
        speed = PAN_SPEED / self.camera.zoom
        self.camera.position = (
            cx + move_x * speed * delta_time,
            cy + move_y * speed * delta_time,
        )

    def on_draw(self) -> None:
        gs = self.game_state
        visible = self._visible_hexes()

        self.clear()
        self.camera.use()
        draw_board(gs.board, self.hex_size, visible_hexes=visible)
        for coord in self.range_preview:
            draw_hex_highlight(coord, self.hex_size, RANGE_PREVIEW_COLOR)
        for coord in self.drag_path:
            draw_hex_highlight(coord, self.hex_size, PATH_HIGHLIGHT_COLOR)
        self._draw_drag_path_line()
        draw_contour(self.contour_segments)
        if visible is None:
            ships_to_draw = gs.ships.values()
        else:
            # Fog of war: the display player's own ships are always shown
            # (see _display_player -- normally current_player, but the
            # watching human's own seat while an AI's turn is playing
            # out); an enemy ship only if currently visible -- see
            # _is_ship_visible for the submerged-submarine and
            # active-battle exceptions.
            ships_to_draw = [
                s for s in gs.ships.values()
                if s.owner == self._display_player() or self._is_ship_visible(s, visible)
            ]
        self._draw_sunk_crossbars()
        draw_ships(ships_to_draw, self.hex_size, current_player=gs.current_player)
        if self.pending_attack_hex is not None:
            draw_hex_highlight(self.pending_attack_hex, self.hex_size, ATTACK_HIGHLIGHT_COLOR)

        self.ui_camera.use()
        if self.game_state.winner is not None:
            if self.replay_writer is not None:
                # A win can occur mid-movement-phase (e.g. an elimination
                # via battle), and the game-over overlay gate below means
                # the player may never get to trigger _end_movement_phase
                # again -- checking here instead, every frame, reliably
                # fires exactly once (write_final no-ops once finalized).
                self.replay_writer.write_final(
                    self.game_state,
                    task_forces=self.ai_policy.task_forces_for(PLAYER_A) + self.ai_policy.task_forces_for(PLAYER_B),
                    posture=self._posture_snapshot(),
                )
            self.ai_policy.close_enemy_belief_store()  # no-op unless --belief was passed; idempotent
            self._draw_game_over_overlay()
            return
        self._draw_hud()
        if self.selected_port is not None:
            self._draw_port_panel()
        if self.sunk_message is not None:
            self._draw_sunk_overlay()
        elif self.pending_attack_hex is not None:
            self._draw_attack_pause_overlay()
        elif self.pending_turn_report:
            self._draw_turn_report()
        elif self.pending_sub_contact:
            self._draw_sub_contact_overlay()
        elif self.active_battle is not None:
            self._draw_battle_banner(self.active_battle)
        elif self._hovered_ship is not None:
            self._draw_hover_tooltip(self._hovered_ship)
        self._draw_toasts()

    def _draw_drag_path_line(self) -> None:
        if len(self.drag_path) < 2:
            return
        points = [axial_to_pixel(coord, self.hex_size) for coord in self.drag_path]
        arcade.draw_line_strip(points, PATH_LINE_COLOR, 3)

    def _draw_sunk_crossbars(self) -> None:
        """A crossbar over every hex in `_sunk_marks` -- see
        `_dismiss_turn_report`/`_record_turn_sunk_hexes` for how that's
        populated and faded. Colored per `_sunk_crossbar_color`, alpha
        scaled by the mark's own current `opacity`. Drawn *before*
        `draw_ships` in `on_draw`, not after -- so a ship now sitting on a
        hex that sank last turn (its own, if it recaptured the spot, or
        an enemy's that moved in afterward) always renders on top of the
        crossbar instead of the crossbar obscuring which ship is there."""
        for hex_coord, mark in self._sunk_marks.items():
            r, g, b = self._sunk_crossbar_color(mark.owners)
            alpha = round(255 * mark.opacity)
            draw_sunk_crossbar(hex_coord, self.hex_size, (r, g, b, alpha))

    def _sunk_crossbar_color(self, owners: set[PlayerId]) -> tuple[int, int, int]:
        """Whichever player's color a sunk-hex crossbar (see
        `_draw_sunk_crossbars`) should use -- that player's own fixed
        `PLAYER_COLORS` entry if only their ship(s) were lost there this
        turn, or `MUTUAL_SUNK_CROSSBAR_COLOR` (black) if hexes sank for
        both sides."""
        if len(owners) > 1:
            return MUTUAL_SUNK_CROSSBAR_COLOR
        return PLAYER_COLORS[next(iter(owners))]

    def _draw_hud(self) -> None:
        gs = self.game_state
        player_label = "Player A" if gs.current_player == PLAYER_A else "Player B"
        lines = [
            f"Turn {gs.turn_number} -- {player_label}'s move    "
            "[Drag a ship] Move    [Esc] Cancel move    "
            "[Enter] End Movement    [T] Toggle hovered submarine",
        ]
        for i, line in enumerate(lines):
            text_obj = self._hud_texts[i]
            text_obj.text = line
            text_obj.y = self.window.height - 22 - i * 20
            text_obj.draw()

    def _port_panel_geometry(self) -> tuple[float, float, float, float]:
        """(left, bottom, width, height) of the port panel in screen space.
        Always sized for exactly one glyph -- a port's build-in-progress is
        a single, automatic, non-interactive value (see tla.production),
        never a queue of choices -- with width floored at
        PORT_PANEL_MIN_WIDTH so the title text isn't cramped."""
        content_width = PORT_PANEL_MARGIN * 2 + PORT_PANEL_ICON_BOX
        width = max(content_width, PORT_PANEL_MIN_WIDTH)
        height = PORT_PANEL_MARGIN + PORT_PANEL_HEADER_HEIGHT + PORT_PANEL_ICON_ROW_HEIGHT
        left = (self.window.width - width) / 2
        return left, PORT_PANEL_MARGIN, width, height

    def _port_panel_slot_center(self) -> tuple[float, float]:
        left, bottom, width, _height = self._port_panel_geometry()
        return left + width / 2, bottom + PORT_PANEL_ICON_ROW_HEIGHT / 2

    def _point_in_port_panel(self, screen_x: float, screen_y: float) -> bool:
        if self.selected_port is None:
            return False
        left, bottom, width, height = self._port_panel_geometry()
        return left <= screen_x <= left + width and bottom <= screen_y <= bottom + height

    def _selected_port_progress(self) -> tuple[ShipKind, int, int] | None:
        """(kind currently being built, points banked toward it, its cost)
        for `self.selected_port`, or None if no port is selected or
        `ProductionConfig.build_order` is empty."""
        if self.selected_port is None:
            return None
        gs = self.game_state
        build_order = gs.config.production.build_order
        if not build_order:
            return None
        progress = gs.players[gs.current_player].port_production.get(self.selected_port)
        index = progress.next_index if progress else 0
        points = progress.points if progress else 0
        kind = build_order[index % len(build_order)]
        return kind, points, gs.config.ship_stats.stats[kind].cost

    def _draw_port_panel(self) -> None:
        gs = self.game_state
        player = gs.current_player

        left, bottom, width, height = self._port_panel_geometry()
        top = bottom + height
        arcade.draw_lbwh_rectangle_filled(left, bottom, width, height, PORT_PANEL_BG_COLOR)
        arcade.draw_lbwh_rectangle_filled(left, top - 4, width, 4, PLAYER_COLORS[player])

        self._port_panel_title_text.text = "Port Production"
        self._port_panel_title_text.x = left + PORT_PANEL_MARGIN
        self._port_panel_title_text.y = top - 20
        self._port_panel_title_text.draw()

        progress = self._selected_port_progress()
        if progress is None:
            return
        kind, points, cost = progress
        half = PORT_PANEL_ICON_BOX / 2
        cx, cy = self._port_panel_slot_center()
        arcade.draw_lbwh_rectangle_outline(cx - half, cy - half, PORT_PANEL_ICON_BOX, PORT_PANEL_ICON_BOX, (100, 100, 100), 1)
        draw_ship_glyph((cx, cy + 9), PORT_PANEL_ICON_HEX_SIZE, kind, PLAYER_COLORS[player])
        self._port_panel_caption_text.text = f"{kind.value.replace('_', ' ').title()}  {points}/{cost}"
        self._port_panel_caption_text.x = cx
        self._port_panel_caption_text.y = cy - half + 5
        self._port_panel_caption_text.draw()

    def _draw_game_over_overlay(self) -> None:
        winner_label = "Player A" if self.game_state.winner == PLAYER_A else "Player B"
        arcade.draw_lbwh_rectangle_filled(
            0, 0, self.window.width, self.window.height, GAME_OVER_BG_COLOR
        )
        self._game_over_text.text = f"{winner_label} wins!"
        self._game_over_text.x = self.window.width / 2
        self._game_over_text.y = self.window.height / 2
        self._game_over_text.draw()

    def _draw_battle_banner(self, battle: ActiveBattle) -> None:
        attacker, defender = battle.attacker, battle.defender
        last_round = battle.rounds[-1]
        stats = self.game_state.config.ship_stats.stats

        # Announced only on the round that made contact -- the defender was
        # hidden by submerged-submarine stealth right up until this attack,
        # so this is the moment it's discovered, not an ongoing label.
        is_sub_contact = (
            len(battle.rounds) == 1
            and defender.kind == ShipKind.SUBMARINE
            and not defender.surfaced
        )

        lines = []
        if is_sub_contact:
            lines.append("SUB CONTACT!")
        lines += [
            f"BATTLE -- {self._ship_label(attacker)} ({attacker.current_hp}/{stats[attacker.kind].hp} HP)"
            f"  vs  {self._ship_label(defender)} ({defender.current_hp}/{stats[defender.kind].hp} HP)",
            f"Round {len(battle.rounds)}: dealt {last_round.damage_to_defender}, "
            f"took {last_round.damage_to_attacker} damage",
            "[Enter] Stay and Fight        [Esc] Retreat",
        ]

        width = 560
        line_height = 24
        height = 16 + line_height * len(lines)
        left = (self.window.width - width) / 2
        top = self.window.height - 40
        accent_color = (230, 180, 40) if is_sub_contact else (200, 60, 60)

        arcade.draw_lbwh_rectangle_filled(left, top - height, width, height, (20, 20, 20, 235))
        arcade.draw_lbwh_rectangle_filled(left, top - 4, width, 4, accent_color)

        for i, line in enumerate(lines):
            text_obj = self._battle_texts[i]
            text_obj.text = line
            text_obj.color = accent_color if is_sub_contact and i == 0 else arcade.color.WHITE
            text_obj.x = left + 12
            text_obj.y = top - 12 - (i + 1) * line_height + 6
            text_obj.draw()

    def _draw_sunk_overlay(self) -> None:
        display_text = f"{self.sunk_message}   (press any key to continue)"
        width = min(self.window.width - 40, max(360, len(display_text) * 9 + 40))
        height = 60
        left = (self.window.width - width) / 2
        top = self.window.height - 40

        arcade.draw_lbwh_rectangle_filled(left, top - height, width, height, SUNK_BG_COLOR)
        arcade.draw_lbwh_rectangle_filled(left, top - 4, width, 4, SUNK_BORDER_COLOR)

        self._sunk_text.text = display_text
        self._sunk_text.x = self.window.width / 2
        self._sunk_text.y = top - height / 2 - 6
        self._sunk_text.draw()

    def _draw_sub_contact_overlay(self) -> None:
        display_text = "Submerged sub encountered!   (press any key to continue)"
        width = min(self.window.width - 40, max(420, len(display_text) * 9 + 40))
        height = 60
        left = (self.window.width - width) / 2
        top = self.window.height - 40

        arcade.draw_lbwh_rectangle_filled(left, top - height, width, height, SUB_CONTACT_BG_COLOR)
        arcade.draw_lbwh_rectangle_filled(left, top - 4, width, 4, SUB_CONTACT_BORDER_COLOR)

        self._sub_contact_text.text = display_text
        self._sub_contact_text.x = self.window.width / 2
        self._sub_contact_text.y = top - height / 2 - 6
        self._sub_contact_text.draw()

    def _draw_attack_pause_overlay(self) -> None:
        display_text = f"{self.pending_attack_message}   (press any key to continue)"
        width = min(self.window.width - 40, max(420, len(display_text) * 9 + 40))
        height = 60
        left = (self.window.width - width) / 2
        top = self.window.height - 40

        arcade.draw_lbwh_rectangle_filled(left, top - height, width, height, ATTACK_BG_COLOR)
        arcade.draw_lbwh_rectangle_filled(left, top - 4, width, 4, ATTACK_BORDER_COLOR)

        self._attack_pause_text.text = display_text
        self._attack_pause_text.x = self.window.width / 2
        self._attack_pause_text.y = top - height / 2 - 6
        self._attack_pause_text.draw()

    def _draw_toasts(self) -> None:
        """Stacked, non-blocking notifications (spotted-ship or
        under-attack) in the top-right corner -- drawn every frame
        regardless of any modal overlay above (except game over, which
        returns before this is ever reached), since neither should
        interrupt whatever else is showing."""
        top = self.window.height - 60  # clears the HUD text at top-left
        right = self.window.width - TOAST_MARGIN
        left = right - TOAST_WIDTH
        shown = self._toasts[: len(self._toast_texts)]
        for i, toast in enumerate(shown):
            box_top = top - i * (TOAST_HEIGHT + 6)
            box_bottom = box_top - TOAST_HEIGHT
            arcade.draw_lbwh_rectangle_filled(left, box_bottom, TOAST_WIDTH, TOAST_HEIGHT, TOAST_BG_COLOR)
            arcade.draw_lbwh_rectangle_filled(left, box_top - 2, TOAST_WIDTH, 2, TOAST_BORDER_COLOR)
            text_obj = self._toast_texts[i]
            text_obj.text = toast.text
            text_obj.x = left + 10
            text_obj.y = box_bottom + TOAST_HEIGHT / 2 - 5
            text_obj.draw()

    def _draw_turn_report(self) -> None:
        """The after-action report for the turn that just ended (both
        players' movement phases) -- HP dealt/taken and any ships lost,
        per player, covering both sides at once across the two sections.
        A sunk ship is drawn as its normal glyph with a red X over it.
        Blocks input until dismissed -- see on_key_press/on_mouse_press
        and _dismiss_turn_report."""
        gs = self.game_state
        width = TURN_REPORT_WIDTH
        height = (
            TURN_REPORT_PADDING * 2
            + TURN_REPORT_TITLE_HEIGHT
            + 2 * (TURN_REPORT_STAT_LINE_HEIGHT + TURN_REPORT_GLYPH_ROW_HEIGHT)
            + TURN_REPORT_FOOTER_HEIGHT
        )
        left = (self.window.width - width) / 2
        top = (self.window.height + height) / 2
        bottom = top - height

        arcade.draw_lbwh_rectangle_filled(left, bottom, width, height, TURN_REPORT_BG_COLOR)
        arcade.draw_lbwh_rectangle_filled(left, top - 4, width, 4, TURN_REPORT_BORDER_COLOR)

        self._turn_report_title_text.text = f"Turn {gs.turn_number} Report"
        self._turn_report_title_text.x = self.window.width / 2
        self._turn_report_title_text.y = top - TURN_REPORT_PADDING - 16
        self._turn_report_title_text.draw()

        cursor_y = top - TURN_REPORT_PADDING - TURN_REPORT_TITLE_HEIGHT
        for i, player in enumerate((PLAYER_A, PLAYER_B)):
            stats = gs.turn_stats[player]
            player_label = "Player A" if player == PLAYER_A else "Player B"

            header_text = self._turn_report_player_texts[i]
            header_text.text = f"{player_label} -- Dealt: {stats.hp_dealt}   Took: {stats.hp_taken}"
            header_text.color = PLAYER_COLORS[player]
            header_text.x = left + TURN_REPORT_PADDING
            header_text.y = cursor_y - TURN_REPORT_STAT_LINE_HEIGHT + 8
            header_text.draw()

            glyph_y = cursor_y - TURN_REPORT_STAT_LINE_HEIGHT - TURN_REPORT_GLYPH_ROW_HEIGHT / 2 + 8
            if stats.ships_lost:
                for j, kind in enumerate(stats.ships_lost):
                    cx = left + TURN_REPORT_PADDING + 30 + j * TURN_REPORT_GLYPH_SPACING
                    draw_ship_glyph((cx, glyph_y), TURN_REPORT_GLYPH_HEX_SIZE, kind, PLAYER_COLORS[player])
                    # Sized to roughly bound the largest hull (~0.5 local
                    # units * SHIP_SCALE from center) at this glyph size --
                    # a fixed multiple of hex_size, not of the individual
                    # ship's own (very different) silhouette size, so a
                    # tiny patrol boat doesn't get lost under an oversized X.
                    half = TURN_REPORT_GLYPH_HEX_SIZE * 0.55
                    arcade.draw_line(cx - half, glyph_y - half, cx + half, glyph_y + half, TURN_REPORT_SUNK_X_COLOR, 3)
                    arcade.draw_line(cx - half, glyph_y + half, cx + half, glyph_y - half, TURN_REPORT_SUNK_X_COLOR, 3)
            else:
                none_text = self._turn_report_none_texts[i]
                none_text.x = left + TURN_REPORT_PADDING
                none_text.y = glyph_y - 5
                none_text.draw()

            cursor_y -= TURN_REPORT_STAT_LINE_HEIGHT + TURN_REPORT_GLYPH_ROW_HEIGHT

        self._turn_report_footer_text.x = self.window.width / 2
        self._turn_report_footer_text.y = bottom + TURN_REPORT_FOOTER_HEIGHT / 2 - 4
        self._turn_report_footer_text.draw()

    def _draw_hover_tooltip(self, ship: Ship) -> None:
        stats = self.game_state.config.ship_stats.stats[ship.kind]
        max_movement = ship.max_movement(stats)

        lines = [
            ship.kind.value.replace("_", " ").title(),
            f"Movement: {ship.movement_remaining}/{max_movement}",
            f"HP: {ship.current_hp}/{stats.hp}",
            f"Damage: {stats.damage}",
        ]
        if ship.kind == ShipKind.SUBMARINE:
            lines.append("Surfaced" if ship.surfaced else "Submerged")
        else:
            lines.append(f"ASW: {stats.asw}")

        height = TOOLTIP_PADDING * 2 + TOOLTIP_LINE_HEIGHT * len(lines)
        mouse_x, mouse_y = self._mouse_screen_pos

        left = mouse_x + TOOLTIP_OFFSET
        if left + TOOLTIP_WIDTH > self.window.width:
            left = mouse_x - TOOLTIP_OFFSET - TOOLTIP_WIDTH
        top = mouse_y + TOOLTIP_OFFSET + height
        if top > self.window.height:
            top = mouse_y - TOOLTIP_OFFSET

        arcade.draw_lbwh_rectangle_filled(left, top - height, TOOLTIP_WIDTH, height, TOOLTIP_BG_COLOR)
        arcade.draw_lbwh_rectangle_filled(left, top - 4, TOOLTIP_WIDTH, 4, PLAYER_COLORS[ship.owner])

        for i, line in enumerate(lines):
            text_obj = self._tooltip_texts[i]
            text_obj.text = line
            text_obj.x = left + TOOLTIP_PADDING
            text_obj.y = top - TOOLTIP_PADDING - (i + 1) * TOOLTIP_LINE_HEIGHT + 4
            text_obj.draw()
