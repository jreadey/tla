# tla
navy strategy game

## Overview

This is a turned-based game where players move their units (ships) on a hexagonal grid.  Each grid hexagon is either land (ships cannot occupy), sea or shore (which ships can occupy).
Each turn of the game involves one side moving their ships (possibly engaging in battle), and production of more ships.  Some shore grid locations represent a players "ports".  These are use for ship production (see production) and are important for winning (see winning).

## Initial State

As the game starts the map will be populated with a number of ships and ports.   Each
players ships are placed near one of their ports.

## Winning

A player wins the game if all his ships are sunk or the opposing player occupies all his ports for a complete turn.

## Turn

A turn consists of the following steps:

* Player A Movement: player A moves his ships (see Movement) possibly engaging in battle with enemy ships
* Player B Movement: player B moves his ships 
* After action report: summary of ships lost, hit points taken and given for each player
* Each player may create new ships (see production)

## Movement

 At most one ship for each player can ge placed in a hexagon at any time.  During a turn, each ship may move a number of sea or shore grid hexagon based on the ship type (see ship types).  

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
  * Submarine: movement (surfaced): 3 movement (submerged): 1, hit points: 4, damage: 4, asw: 0, cost: 4
  * Patrol boat: movement 6, hit points: 2, damage 1: asw: 1: cost 1

## Battle

During movement if a ship enters a grid occupied by an enemy ship they engage in mutual attacks.  Each ships hit points will decrease (with a minimum of zero) based on the damage number of the opposing ship.  If the number of hit points is zero, the ship is "sunk" and removed from the game.  After the exchange of damage, the ship that moved into the occupied hexagon can stay or retreat (return to the hexagon it entered from).  If the ship stays, another exchange of damage will begin (continuing until either one or both of the ships are sunk, or the player's ship withdraws).  Engaging costs one movement point, the same as moving into an empty hex -- a retreat is free, so an attack followed by a retreat still only costs that one point. If the ship has movement left afterward (win and continue, or retreat), it can keep moving that turn.

One special rule, for each aircraft carrier that is within one hexagon of the hexagon 
where the battle is taking place, one damage point is added to the attack for the ship that is on the same side as the carrier.

## Production

Production is automatic and organized per port. Clicking an empty friendly port opens that port's own production panel, showing its current queue as a row of ship glyphs plus an "Order" button; hovering a queued glyph shows its ship type and build state (e.g. "8/10"), and pressing Order opens a picker of the six ship types (with costs) to add to that port's queue, in any quantity. Each turn's production budget (20 points by default) is split evenly across the ports that currently have something queued and aren't occupied by a ship (of either side); a port banks its share and, once it has enough to cover the cost of the order at the front of its queue, the new ship appears there and any leftover points carry over toward the port's next order. A port occupied by either side's ship gets no share that turn -- its points are simply redistributed to the player's other eligible ports instead of being wasted -- and if the occupant is an enemy ship, that port's entire queue and banked points are lost outright, resuming from empty once the port is free again.

## Development

The game is written in Python (version 3.12) using Python Arcade.  The map will be created using a Perlin noise algorithm.  Ports and ships will be placed randomly. 

The game can be played in two-player mode (humans controlling each side) or one-player were the opposing side is controlled by an AI.
  

