# tla
navy strategy game

## Overview

This is a turned-based game where players move their units (ships) on a hexagonal grid.  Each grid hexagon is either land (ships cannot occupy), sea or shore (which ships can occupy).
Each turn of the game involves one side moving their ships (possibly engaging in battle), and production of more ships.  Some shore grid locations represent a players "ports".  These are use for ship production (see production) and are important for winning (see winning).

## Initial State

As the game starts the map will be populated with a number of ships and ports.   Each
players ships are placed near one of their ports.

## Winning

A player wins the game the instant their opponent's entire fleet is sunk, or the instant one player controls every port on the map -- their own plus every one they've captured from the other side (see Production above: a captured port stays captured, and can be built from, until the other side takes it back).

## Turn

A turn consists of the following steps:

* Player A Movement: player A moves his ships (see Movement) possibly engaging in battle with enemy ships
* Player B Movement: player B moves his ships 
* After action report: summary of ships lost, hit points taken and given for each player
* Each player may create new ships (see production)

## Movement

 At most one ship for each player can be placed in a hexagon at the end of a turn. During a turn, each ship may move a number of sea or shore grid hexagons based on the ship type (see ship types). A ship may pass *through* a hexagon occupied by a friendly ship without stopping there, as long as it still has at least 2 movement points left when it enters that hexagon (so it always has enough left to continue on to a legal hexagon beyond it, rather than being stranded there).

If a ship moves into a hexagon occupied by an enemy ship, the ship's movement for this turn immediately stops and it engages in battle with the opposing ship (see battle).

A submarine can be in a state of either surfaced or submerged.  During a submarines movement, it first can optionally surface (if submerged), or submerge (if surfaced).  
After movement is concluded (any any potential battle completed), the submarine can again surface or submerge.

## Ship Types

There are a number of different ship types that have different characteristics:

  * movement: maximum number of hexagons a ship move move in a turn
  * hit points: amount of damage a ship can take before it is sunk
  * damage: amount of damage the ship can inflict on a enemy ship (other than submerged submarines) during a battle
  * asw: amount of damage a ship can inflict on submerged enemy submarines
  * cost: number pf production points needed to create a new ship of this type

The ship types are:

  * Battleships: movement: 4, hit points: 12,  damage: 4, asw: 0, cost: 10
  * Aircraft Carrier: movement: 4, hit points: 7, damage 2, asw: 0, cost: 10
  * Cruisers: movement: 4,  hit points: 8, damage: 4, asw: 2, cost: 7
  * Destroyer: movement: 4, hit points: 6, damage: 2, asw: 2, cost: 4
  * Submarine: movement (surfaced): 3 movement (submerged): 1, hit points: 4, damage: 4, asw: 1, cost: 4
  * Patrol boat: movement 6, hit points: 2, damage 1: asw: 1: cost 1

## Battle

During movement if a ship enters a grid occupied by an enemy ship they engage in mutual attacks.  Each ships hit points will decrease (with a minimum of zero) based on the damage number of the opposing ship.  If the number of hit points is zero, the ship is "sunk" and removed from the game.  After the exchange of damage, the ship that moved into the occupied hexagon can stay or retreat (return to wherever it stopped just before entering that hexagon).  If the ship stays, another exchange of damage will begin (continuing until either one or both of the ships are sunk, or the player's ship withdraws).  Engaging costs the same movement as covering that distance normally would -- usually just the 1 point for the final step into the enemy's hexagon, but if a friendly ship passed through along the way happens to sit in the one hexagon adjacent to the target, the attacker can't literally stop there alongside it, so it stops one hexagon further back instead and the whole remaining stretch is charged as the attack. A retreat is always free, returning to exactly wherever the attacker stopped. If the ship has movement left afterward (win and continue, or retreat), it can keep moving that turn.

One special rule, for each aircraft carrier that is within one hexagon of the hexagon 
where the battle is taking place, one damage point is added to the attack for the ship that is on the same side as the carrier. This air-cover bonus only benefits the larger surface ships -- carriers, battleships, cruisers, and destroyers; a submarine or patrol boat gets no bonus regardless of how many friendly carriers are nearby.

## Production

Production is automatic and organized per port. Clicking an empty friendly port opens that port's own production panel, showing its current queue as a row of ship glyphs plus an "Order" button; hovering a queued glyph shows its ship type and build state (e.g. "8/10"), and pressing Order opens a picker of the six ship types (with costs) to add to that port's queue, in any quantity. Every turn, each port a player currently controls (their own plus any of the opponent's they've captured) and that isn't occupied by a ship independently earns 5 production points (configurable) -- not a shared budget split across ports, so controlling more ports means more total production, not a thinner split of a fixed amount. A port banks its points and, once they cover the cost of the order at the front of its queue, the new ship appears there and any leftover carries toward the port's next order; a port occupied by either side's ship simply earns nothing that turn. If the occupant is an enemy ship, that port's entire queue and banked points are also lost outright, resuming from empty once the port is free again.

A port an enemy has captured can be built from by whoever currently holds it -- production follows the flip. Losing control of a port, whether to the original owner retaking it or a second capture by the other side, wipes whatever the previous controller had queued there, the same as an initial capture does.

## Fog of War

Fog of war is an optional rule (off by default, enabled via the `fow` config). When enabled, a player can't see the enemy's ships except where they currently have vision: any hex within 1 cell of one of their own ships, or within 4 cells of a port they control or one of their own aircraft carriers (both configurable). The map itself -- terrain, coastline, and who controls which port -- is always fully visible either way; fog of war only ever hides enemy ship positions. During a player's turn, the hexes currently within their vision are shown in a lighter shade so the extent of their vision is visible at a glance.

A submerged enemy submarine stays hidden even within an otherwise-visible hex -- normal detection doesn't spot it. The only way to find one is to sail directly into it and trigger a battle; only then is it revealed, for the duration of that engagement, with a "SUB CONTACT!" banner announcing the moment of discovery.

Drawing out a move never gives away a hidden submarine's location. While dragging, the preview behaves as though a hidden sub weren't there at all -- it doesn't block movement or narrow the highlighted range -- so its presence can't be inferred from where the drag refuses to extend. If the route you actually release on runs through a hidden sub's hex, your ship stops there and engages it, even if you'd aimed further; it just couldn't see the danger coming.

## Development

The game is written in Python (version 3.12) using Python Arcade.  The map will be created using a Perlin noise algorithm.  Ports and ships will be placed randomly. 

The game can be played in two-player mode (humans controlling each side) or one-player were the opposing side is controlled by an AI.
  

