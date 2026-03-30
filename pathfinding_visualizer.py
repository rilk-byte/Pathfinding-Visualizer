"""
Pathfinding Visualizer — Python + Tkinter
Algorithms: BFS · DFS · Dijkstra · A*
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import heapq
from collections import deque
import time

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
ROWS, COLS = 28, 50
CELL = 22
ANIM_DELAY = 18   # ms between visited-cell renders
PATH_DELAY = 30   # ms between path-cell renders

# Colour palette (dark terminal aesthetic)
BG        = "#0d1117"
PANEL_BG  = "#161b22"
GRID_BG   = "#0d1117"
BORDER    = "#30363d"
EMPTY     = "#0d1117"
WALL      = "#58a6ff"
START_CLR = "#3fb950"
END_CLR   = "#f85149"
VISITED   = "#1f6feb"
FRONTIER  = "#388bfd"
PATH_CLR  = "#f0e68c"
TEXT_CLR  = "#e6edf3"
MUTED     = "#8b949e"
ACCENT    = "#58a6ff"
BTN_BG    = "#21262d"
BTN_HOV   = "#30363d"
BTN_ACT   = "#388bfd"
WEIGHT_CLR= "#d29922"

# Cell states
EMPTY_S  = 0
WALL_S   = 1
START_S  = 2
END_S    = 3
VISITED_S= 4
PATH_S   = 5
WEIGHT_S = 6   # for Dijkstra demo

ALGO_INFO = {
    "BFS":      ("Breadth-First Search",  "Guarantees shortest path on unweighted grids. Explores all neighbours level by level."),
    "DFS":      ("Depth-First Search",    "Not optimal — dives deep before backtracking. Fast but may find long paths."),
    "Dijkstra": ("Dijkstra's Algorithm",  "Optimal weighted pathfinding. Weight cells cost 5× more to traverse."),
    "A*":       ("A* Search",             "Uses a heuristic (Manhattan distance) to reach the goal faster than Dijkstra."),
}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def neighbours(r, c):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            yield nr, nc

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


# ──────────────────────────────────────────────
# ALGORITHMS  — all return (visited_order, path)
# ──────────────────────────────────────────────
def bfs(grid, start, end):
    queue = deque([start])
    came_from = {start: None}
    visited = []
    while queue:
        cur = queue.popleft()
        if cur == end:
            break
        visited.append(cur)
        for nb in neighbours(*cur):
            if nb not in came_from and grid[nb[0]][nb[1]] != WALL_S:
                came_from[nb] = cur
                queue.append(nb)
    return visited, _reconstruct(came_from, start, end)


def dfs(grid, start, end):
    stack = [start]
    came_from = {start: None}
    visited = []
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.append(cur)
        if cur == end:
            break
        for nb in neighbours(*cur):
            if nb not in came_from and grid[nb[0]][nb[1]] != WALL_S:
                came_from[nb] = cur
                stack.append(nb)
    return visited, _reconstruct(came_from, start, end)


def dijkstra(grid, start, end):
    dist = {start: 0}
    came_from = {start: None}
    heap = [(0, start)]
    visited = []
    while heap:
        cost, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.append(cur)
        if cur == end:
            break
        for nb in neighbours(*cur):
            cell = grid[nb[0]][nb[1]]
            if cell == WALL_S:
                continue
            w = 5 if cell == WEIGHT_S else 1
            nc = cost + w
            if nc < dist.get(nb, float('inf')):
                dist[nb] = nc
                came_from[nb] = cur
                heapq.heappush(heap, (nc, nb))
    return visited, _reconstruct(came_from, start, end)


def astar(grid, start, end):
    came_from = {start: None}
    g = {start: 0}
    f = {start: manhattan(start, end)}
    heap = [(f[start], start)]
    visited = []
    while heap:
        _, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.append(cur)
        if cur == end:
            break
        for nb in neighbours(*cur):
            cell = grid[nb[0]][nb[1]]
            if cell == WALL_S:
                continue
            w = 5 if cell == WEIGHT_S else 1
            ng = g[cur] + w
            if ng < g.get(nb, float('inf')):
                g[nb] = ng
                came_from[nb] = cur
                heapq.heappush(heap, (ng + manhattan(nb, end), nb))
    return visited, _reconstruct(came_from, start, end)


def _reconstruct(came_from, start, end):
    if end not in came_from:
        return []
    path, cur = [], end
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
class PathfinderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pathfinding Visualizer")
        self.configure(bg=BG)
        self.resizable(False, False)

        # State
        self.grid   = [[EMPTY_S]*COLS for _ in range(ROWS)]
        self.start  = None
        self.end    = None
        self.mode   = tk.StringVar(value="wall")   # wall | start | end | weight | erase
        self.algo   = tk.StringVar(value="A*")
        self.speed  = tk.IntVar(value=ANIM_DELAY)
        self.running= False
        self._after_ids = []

        self._build_ui()
        self._draw_full_grid()

    # ── UI CONSTRUCTION ──────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG, pady=8)
        hdr.pack(fill="x", padx=16)
        tk.Label(hdr, text="⬡ PATHFINDING VISUALIZER", bg=BG, fg=ACCENT,
                 font=("Courier New", 16, "bold")).pack(side="left")
        tk.Label(hdr, text="BFS · DFS · Dijkstra · A*", bg=BG, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=12, pady=4)

        # Main container
        main = tk.Frame(self, bg=BG)
        main.pack(padx=14, pady=4)

        # Canvas
        cw = COLS * CELL + 1
        ch = ROWS * CELL + 1
        self.canvas = tk.Canvas(main, width=cw, height=ch, bg=GRID_BG,
                                highlightthickness=1, highlightbackground=BORDER, cursor="crosshair")
        self.canvas.pack(side="left")
        self.canvas.bind("<Button-1>",        self._on_click)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<Button-3>",        self._on_right_click)
        self.canvas.bind("<B3-Motion>",       self._on_right_drag)

        # Side panel
        panel = tk.Frame(main, bg=PANEL_BG, width=220, padx=14, pady=14,
                         highlightthickness=1, highlightbackground=BORDER)
        panel.pack(side="left", fill="y", padx=(10,0))
        panel.pack_propagate(False)
        self._build_panel(panel)

        # Status bar
        self.status_var = tk.StringVar(value="Ready — draw walls and set Start/End")
        sb = tk.Frame(self, bg=PANEL_BG, pady=6,
                      highlightthickness=1, highlightbackground=BORDER)
        sb.pack(fill="x", padx=14, pady=(4,10))
        tk.Label(sb, textvariable=self.status_var, bg=PANEL_BG, fg=TEXT_CLR,
                 font=("Courier New", 9), anchor="w").pack(side="left", padx=10)
        self.stat_right = tk.Label(sb, text="", bg=PANEL_BG, fg=MUTED,
                                   font=("Courier New", 9))
        self.stat_right.pack(side="right", padx=10)

    def _build_panel(self, p):
        def section(label):
            tk.Label(p, text=label, bg=PANEL_BG, fg=MUTED,
                     font=("Courier New", 7, "bold")).pack(anchor="w", pady=(10,2))

        # Algorithm picker
        section("ALGORITHM")
        for algo in ["BFS", "DFS", "Dijkstra", "A*"]:
            self._radio(p, algo, self.algo, command=self._update_algo_info)

        # Info box
        self.info_title = tk.Label(p, text="", bg=PANEL_BG, fg=ACCENT,
                                   font=("Courier New", 8, "bold"), wraplength=190, justify="left")
        self.info_title.pack(anchor="w", pady=(6,0))
        self.info_body  = tk.Label(p, text="", bg=PANEL_BG, fg=MUTED,
                                   font=("Courier New", 7), wraplength=190, justify="left")
        self.info_body.pack(anchor="w")
        self._update_algo_info()

        # Draw mode
        section("DRAW MODE")
        modes = [("🟩 Set Start",  "start"),
                 ("🟥 Set End",    "end"),
                 ("⬛ Wall",       "wall"),
                 ("🟨 Weight 5", "weight"),
                 ("✏️  Erase",     "erase")]
        for label, val in modes:
            self._radio(p, label, self.mode, val=val)

        # Speed
        section("ANIMATION SPEED")
        spd_frame = tk.Frame(p, bg=PANEL_BG)
        spd_frame.pack(fill="x")
        tk.Label(spd_frame, text="Fast", bg=PANEL_BG, fg=MUTED,
                 font=("Courier New", 7)).pack(side="left")
        tk.Scale(spd_frame, from_=2, to=80, orient="horizontal",
                 variable=self.speed, bg=PANEL_BG, fg=TEXT_CLR,
                 troughcolor=BTN_BG, highlightthickness=0,
                 showvalue=False, length=120).pack(side="left", padx=4)
        tk.Label(spd_frame, text="Slow", bg=PANEL_BG, fg=MUTED,
                 font=("Courier New", 7)).pack(side="left")

        # Buttons
        section("CONTROLS")
        self.run_btn = self._button(p, "▶  VISUALIZE", self._run, bg=BTN_ACT)
        self._button(p, "↺  RESET PATH",   self._reset_path)
        self._button(p, "⊘  CLEAR GRID",   self._clear_grid)
        self._button(p, "≡  RANDOM MAZE",  self._random_maze)

        # Legend
        section("LEGEND")
        legends = [(START_CLR, "Start"), (END_CLR, "End"), (WALL, "Wall"),
                   (WEIGHT_CLR, "Weight ×5"), (VISITED, "Visited"),
                   (PATH_CLR, "Shortest Path")]
        for clr, lbl in legends:
            row = tk.Frame(p, bg=PANEL_BG)
            row.pack(anchor="w", pady=1)
            tk.Canvas(row, width=12, height=12, bg=clr,
                      highlightthickness=0).pack(side="left", padx=(0,6))
            tk.Label(row, text=lbl, bg=PANEL_BG, fg=MUTED,
                     font=("Courier New", 7)).pack(side="left")

    def _radio(self, parent, label, var, val=None, command=None):
        if val is None:
            val = label
        rb = tk.Radiobutton(parent, text=label, variable=var, value=val,
                            bg=PANEL_BG, fg=TEXT_CLR, selectcolor=PANEL_BG,
                            activebackground=PANEL_BG, activeforeground=ACCENT,
                            font=("Courier New", 9), indicatoron=False,
                            relief="flat", bd=0, cursor="hand2",
                            command=command,
                            pady=3, padx=8)
        rb.pack(fill="x", pady=1)
        rb.bind("<Enter>", lambda e: rb.configure(fg=ACCENT))
        rb.bind("<Leave>", lambda e: rb.configure(fg=TEXT_CLR if var.get()!=val else ACCENT))
        return rb

    def _button(self, parent, label, cmd, bg=BTN_BG):
        btn = tk.Button(parent, text=label, command=cmd, bg=bg, fg=TEXT_CLR,
                        activebackground=BTN_HOV, activeforeground=ACCENT,
                        font=("Courier New", 9, "bold"), relief="flat",
                        cursor="hand2", pady=6, bd=0)
        btn.pack(fill="x", pady=2)
        btn.bind("<Enter>", lambda e: btn.configure(bg=BTN_HOV))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg))
        return btn

    # ── GRID DRAWING ─────────────────────────────
    def _draw_full_grid(self):
        self.canvas.delete("all")
        self.cell_ids = {}
        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = c*CELL, r*CELL
                x2, y2 = x1+CELL, y1+CELL
                cid = self.canvas.create_rectangle(
                    x1+1, y1+1, x2, y2,
                    fill=self._cell_color(r,c), outline="", tags=f"cell_{r}_{c}")
                self.cell_ids[(r,c)] = cid
        # grid lines
        for r in range(ROWS+1):
            self.canvas.create_line(0, r*CELL, COLS*CELL, r*CELL, fill=BORDER, width=1)
        for c in range(COLS+1):
            self.canvas.create_line(c*CELL, 0, c*CELL, ROWS*CELL, fill=BORDER, width=1)

    def _cell_color(self, r, c):
        s = self.grid[r][c]
        if (r,c) == self.start:  return START_CLR
        if (r,c) == self.end:    return END_CLR
        return {EMPTY_S:EMPTY, WALL_S:WALL, VISITED_S:VISITED,
                PATH_S:PATH_CLR, WEIGHT_S:WEIGHT_CLR}.get(s, EMPTY)

    def _paint_cell(self, r, c, color=None):
        cid = self.cell_ids.get((r,c))
        if cid:
            clr = color or self._cell_color(r,c)
            self.canvas.itemconfig(cid, fill=clr)

    # ── MOUSE EVENTS ─────────────────────────────
    def _cell_at(self, event):
        c = event.x // CELL
        r = event.y // CELL
        if 0 <= r < ROWS and 0 <= c < COLS:
            return r, c
        return None

    def _on_click(self, event):
        if self.running: return
        cell = self._cell_at(event)
        if cell: self._apply_mode(*cell)

    def _on_drag(self, event):
        if self.running: return
        cell = self._cell_at(event)
        if cell and self.mode.get() in ("wall","weight","erase"):
            self._apply_mode(*cell)

    def _on_right_click(self, event):
        if self.running: return
        cell = self._cell_at(event)
        if cell: self._erase_cell(*cell)

    def _on_right_drag(self, event):
        if self.running: return
        cell = self._cell_at(event)
        if cell: self._erase_cell(*cell)

    def _apply_mode(self, r, c):
        m = self.mode.get()
        if m == "start":
            if self.start:
                old = self.start
                self.grid[old[0]][old[1]] = EMPTY_S
                self._paint_cell(*old)
            self.start = (r,c)
            self.grid[r][c] = START_S
            self._paint_cell(r,c, START_CLR)
        elif m == "end":
            if self.end:
                old = self.end
                self.grid[old[0]][old[1]] = EMPTY_S
                self._paint_cell(*old)
            self.end = (r,c)
            self.grid[r][c] = END_S
            self._paint_cell(r,c, END_CLR)
        elif m == "wall":
            if (r,c) not in (self.start, self.end):
                self.grid[r][c] = WALL_S
                self._paint_cell(r,c, WALL)
        elif m == "weight":
            if (r,c) not in (self.start, self.end):
                self.grid[r][c] = WEIGHT_S
                self._paint_cell(r,c, WEIGHT_CLR)
        elif m == "erase":
            self._erase_cell(r,c)

    def _erase_cell(self, r, c):
        if (r,c) == self.start: self.start = None
        if (r,c) == self.end:   self.end   = None
        self.grid[r][c] = EMPTY_S
        self._paint_cell(r,c, EMPTY)

    # ── CONTROLS ─────────────────────────────────
    def _reset_path(self):
        self._cancel_anims()
        self.running = False
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] in (VISITED_S, PATH_S):
                    self.grid[r][c] = EMPTY_S
                    self._paint_cell(r,c, EMPTY)
        self.status_var.set("Path cleared — ready to visualize again")
        self.stat_right.config(text="")

    def _clear_grid(self):
        self._cancel_anims()
        self.running = False
        self.start = None
        self.end   = None
        self.grid  = [[EMPTY_S]*COLS for _ in range(ROWS)]
        self._draw_full_grid()
        self.status_var.set("Grid cleared")
        self.stat_right.config(text="")

    def _random_maze(self):
        import random
        self._clear_grid()
        for r in range(ROWS):
            for c in range(COLS):
                if random.random() < 0.30:
                    self.grid[r][c] = WALL_S
                    self._paint_cell(r,c, WALL)
        # random start/end far apart
        self.start = (random.randint(0, ROWS-1), random.randint(0, COLS//4))
        self.end   = (random.randint(0, ROWS-1), random.randint(3*COLS//4, COLS-1))
        for pos in (self.start, self.end):
            self.grid[pos[0]][pos[1]] = EMPTY_S
        self._paint_cell(*self.start, START_CLR)
        self._paint_cell(*self.end,   END_CLR)
        self.status_var.set("Random maze generated — click Visualize!")

    def _update_algo_info(self):
        a = self.algo.get()
        title, body = ALGO_INFO.get(a, ("",""))
        self.info_title.config(text=title)
        self.info_body.config(text=body)

    def _cancel_anims(self):
        for aid in self._after_ids:
            self.after_cancel(aid)
        self._after_ids.clear()

    # ── RUN ──────────────────────────────────────
    def _run(self):
        if self.running:
            return
        if not self.start or not self.end:
            self.status_var.set("⚠  Please set both Start and End points")
            return
        self._reset_path()
        self.running = True
        self.run_btn.config(state="disabled")

        algo = self.algo.get()
        fn   = {"BFS": bfs, "DFS": dfs, "Dijkstra": dijkstra, "A*": astar}[algo]

        t0 = time.perf_counter()
        visited, path = fn(self.grid, self.start, self.end)
        elapsed = time.perf_counter() - t0

        self.status_var.set(f"Running {algo}…")
        self._animate_visited(visited, path, elapsed)

    def _animate_visited(self, visited, path, elapsed):
        delay = self.speed.get()

        def step(i):
            if i < len(visited):
                r,c = visited[i]
                if (r,c) not in (self.start, self.end):
                    self.grid[r][c] = VISITED_S
                    self._paint_cell(r,c, VISITED)
                aid = self.after(delay, lambda: step(i+1))
                self._after_ids.append(aid)
            else:
                self._animate_path(path, elapsed)

        step(0)

    def _animate_path(self, path, elapsed):
        delay = self.speed.get() + PATH_DELAY

        if not path:
            self.status_var.set("⚠  No path found! Try removing some walls.")
            self._finish()
            return

        def step(i):
            if i < len(path):
                r,c = path[i]
                if (r,c) not in (self.start, self.end):
                    self.grid[r][c] = PATH_S
                    self._paint_cell(r,c, PATH_CLR)
                aid = self.after(delay, lambda: step(i+1))
                self._after_ids.append(aid)
            else:
                algo = self.algo.get()
                self.status_var.set(
                    f"✓ {algo} complete — path length: {len(path)-1} steps")
                self.stat_right.config(
                    text=f"visited: {sum(1 for r in self.grid for s in r if s==VISITED_S)} cells  |  "
                         f"calc: {elapsed*1000:.1f}ms")
                self._finish()

        step(0)

    def _finish(self):
        self.running = False
        self.run_btn.config(state="normal")


# ──────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = PathfinderApp()
    app.mainloop()
