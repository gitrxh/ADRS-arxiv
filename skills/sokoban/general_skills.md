### EXPERT SOKOBAN STRATEGY GUIDE ###

You are solving a Sokoban puzzle. The grid uses these symbols:
- `#` = wall (impassable)
- `_` = floor (walkable)
- `O` = target (where boxes must go)
- `X` = box (push it onto a target)
- `P` = player (you)
- `√` = box already on target (success!)
- `S` = player standing on a target

Your goal: push every X onto an O. You win when all boxes show √.

## CRITICAL RULES

1. You can only PUSH boxes, never pull. Stand on the opposite side from where you want the box to go.
2. To push a box LEFT → stand to its RIGHT → move LEFT.
3. To push a box UP → stand BELOW it → move UP.
4. NEVER push a box into a corner — it gets permanently stuck.
5. NEVER push two boxes side-by-side against a wall.

## STEP-BY-STEP SOLVING METHOD

For EVERY move, follow this checklist:
1. READ the grid row by row. Write out coordinates: "Box at (row,col), Target at (row,col), Player at (row,col)"
2. PLAN: Which direction must the box move? (toward target)
3. POSITION: Where must I stand to push it that way?
4. CHECK: Will this push trap the box? (corners = death)
5. ACT: Choose one of [up, down, left, right]

## COMMON PATTERNS

Pattern A — Direct Push:
  Box and target on same row/column → just push straight.
  Example: `O _ X P` → push LEFT three times → done.

Pattern B — L-Shape:
  Box needs to go LEFT then UP (or any two directions).
  Step 1: Push box in first direction until aligned with target column/row.
  Step 2: Reposition yourself to the new push side.
  Step 3: Push box in second direction to target.

Pattern C — Avoiding Walls:
  If target is along a wall, push the box PARALLEL to the wall first,
  then push it INTO the wall-side target. Never push perpendicular to
  a wall unless the target is right there.

## WORKED EXAMPLES

Example 1: Simple horizontal push
Grid: ['# # # # # #', '# O _ X P #', '# _ _ _ _ #', '# # # # # #']
Analysis: Target O at (1,1), Box X at (1,3), Player P at (1,4).
  Box must go LEFT 2 steps. Player is already RIGHT of box.
  → Action: LEFT (pushes box from (1,3) to (1,2))
  → Action: LEFT (pushes box from (1,2) to (1,1) = target!)

Example 2: L-shape maneuver
Grid: ['# # # # # #', '# O _ _ _ #', '# _ X _ _ #', '# _ _ P _ #', '# # # # # #']
Analysis: Target at (1,1), Box at (2,2), Player at (3,3).
  Box needs LEFT 1 and UP 1. Push LEFT first:
  → Move to (2,3) — RIGHT of box → push LEFT → box at (2,1)
  → Move to (3,1) — BELOW box → push UP → box at (1,1) = target!

Example 3: Dangerous corner
Grid: ['# # # # # #', '# O _ _ _ #', '# _ _ _ X #', '# _ _ P _ #', '# # # # # #']
  Box at (2,4) near right wall. If pushed UP → (1,4) = corner with wall → STUCK!
  Must push LEFT first to escape the wall, then UP.

## KEY INSIGHT
Always push boxes AWAY from walls first, then toward the target. Pushing toward a wall without the target being there is almost always a mistake.