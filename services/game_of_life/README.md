## Game of Life

### Demo video link
https://www.youtube.com/watch?v=lvkF7i6pmR8

### Introduction

We use 9 segments to simulate **Conway’s Game of Life**. Each segment uses its first 4096 bytes to represent a 64×64 page. These 9 pages are arranged into a 3×3 grid, forming the complete simulation space:

```
segment_0[:4096] | segment_1[:4096] | segment_2[:4096]
------------------------------------------------------
segment_3[:4096] | segment_4[:4096] | segment_5[:4096]
------------------------------------------------------
segment_6[:4096] | segment_7[:4096] | segment_8[:4096]
```

### Initialization

The simulation is initialized by placing `n` **gliders** into the grid. Each glider moves diagonally—toward the bottom-left—across steps, following the standard Game of Life rules.

### Step Details

To experiment with the Game of Life interactively, visit:  
🔗 [https://playgameoflife.com/](https://playgameoflife.com/)  
This website allows you to configure any cell manually and watch the simulation evolve step by step.

The program updates the simulation by `total_execution_steps` steps at a time. The `total_execution_steps` parameter can be set in the payload.  

Below is a **schematic output** of our simulation with a single glider and `total_execution_steps = 10`.
> ⚠️ **Note**: These diagrams are for illustration only and do not reflect actual segment or byte-level data.

---

#### Step 0

```
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬛⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬛⬜⬜⬜⬜⬜  
⬜⬜⬛⬛⬛⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
```

...

#### Step 10

```
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬛⬜⬜⬜  
⬜⬜⬜⬜⬛⬜⬛⬜⬜⬜  
⬜⬜⬜⬜⬜⬛⬛⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
```

...

#### Step 20

```
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬛⬜  
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬛  
⬜⬜⬜⬜⬜⬜⬜⬛⬛⬛  
```
...

---

### Import / Export Segments

- **Step 0**:  
  We initialize the entire grid with `n` gliders, perform one simulation step, and export all 9 segments.

- **Step ≥ 1**:  
  We import the 9 segments from the previous step, perform one simulation step, and then export all 9 segments again.
