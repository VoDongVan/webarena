# browser_env/ — Browser Environment

The browser automation layer. Wraps a real Chromium browser (via Playwright) as a Gymnasium-compatible environment.

---

## File Map

| File | Role |
|---|---|
| `envs.py` | Synchronous Gymnasium environment (`ScriptBrowserEnv`) |
| `async_envs.py` | Async variant (image observations only) |
| `actions.py` | 18 action types, execution engine, action parsing |
| `processors.py` | DOM / accessibility tree → text observation |
| `auto_login.py` | Cookie-based pre-authentication |
| `constants.py` | ARIA roles, character encodings, limits |
| `env_config.py` | Site URLs (from env vars) + hardcoded test accounts |
| `utils.py` | Type aliases (`Observation`, `DOMNode`, `DetachedPage`) |
| `helper_functions.py` | HTML trajectory renderer (`RenderHelper`) |
| `trajectory.py` | `Trajectory = list[StateInfo \| Action]` |

---

## ScriptBrowserEnv (`envs.py`)

A `gymnasium.Env[dict, Action]` that controls a real browser.

### Constructor Options

```python
ScriptBrowserEnv(
    max_page_length=8192,       # max HTML/tree text length
    headless=True,              # show browser window?
    slow_mo=0,                  # slow down actions (ms), useful for debugging
    observation_type="accessibility_tree",  # "html" | "accessibility_tree" | "image"
    current_viewport_only=False,  # filter out off-screen DOM nodes
    viewport_size={"width": 1280, "height": 720},
    save_trace_enabled=False,   # save Playwright trace zip
    sleep_after_execution=0.0,  # wait after each action for page to settle
)
```

### Key Methods

**`setup(config_file)`**
- Launches Chromium, creates browser context
- Loads `storage_state` JSON (cookies) if present → agent starts pre-authenticated
- Opens start URL(s); multiple tabs separated by `|AND|` in the URL string
- Enables a CDP (Chrome DevTools Protocol) session for accessibility tree access

**`reset(seed, options)` → `(obs_dict, info_dict)`**
- Reloads the page to the task's start state
- Returns initial observation

**`step(action)` → `(obs, reward, terminated, truncated, info)`**
- Executes one action in the browser via `execute_action()`
- `reward = 1.0` if no exception was raised, `0.0` otherwise
- `terminated` and `truncated` are always `False` (no built-in episode end)
- `info` contains full page HTML and URL

**`save_trace(path)`** — saves Playwright trace for debugging

### Observation Dict

```python
{
    "text":  str,                  # accessibility tree or HTML string (up to 8192 chars)
    "image": np.ndarray,           # screenshot, shape (720, 1280, 4) RGBA
}
```

---

## Action Space (`actions.py`)

### ActionTypes (IntEnum) — 18 types

| Category | Action | ID | Notes |
|---|---|---|---|
| Low-level | `SCROLL` | 1 | `direction`: "up"/"down" |
| | `KEY_PRESS` | 2 | `key_comb`: "Control+a" |
| | `MOUSE_CLICK` | 3 | `coords`: normalized [0,1] |
| | `KEYBOARD_TYPE` | 4 | `text`: list of char IDs |
| | `MOUSE_HOVER` | 5 | `coords`: normalized [0,1] |
| Element-based | `CLICK` | 6 | by `element_id` or `pw_code` |
| | `TYPE` | 7 | by `element_id` or `pw_code` |
| | `HOVER` | 8 | by `element_id` or `pw_code` |
| | `CHECK` | 15 | checkbox |
| | `SELECT_OPTION` | 16 | dropdown |
| Navigation | `PAGE_FOCUS` | 9 | switch tab |
| | `NEW_TAB` | 10 | |
| | `GO_BACK` | 11 | |
| | `GO_FORWARD` | 12 | |
| | `GOTO_URL` | 13 | `url` field |
| | `PAGE_CLOSE` | 14 | |
| Terminal | `STOP` | 17 | `answer` field = final answer |
| No-op | `NONE` | 0 | parse failure fallback |

### Action TypedDict

```python
Action = TypedDict {
    action_type: int,        # ActionTypes enum
    element_id: str,         # e.g. "42" — from accessibility tree
    coords: ndarray[2],      # normalized [0,1] mouse position
    text: list[int],         # char IDs for typing (max 64)
    key_comb: str,           # e.g. "Control+a"
    url: str,                # for GOTO_URL
    direction: str,          # "up"/"down" for SCROLL
    pw_code: str,            # raw Playwright code string
    answer: str,             # final answer for STOP
    page_number: int,        # tab index for PAGE_FOCUS
    raw_prediction: str,     # original LLM output (stored by agent)
}
```

### Element Lookup Priority (CLICK / TYPE / HOVER)

1. `element_id` — direct lookup via `obs_nodes_info` (fastest)
2. `element_role + element_name` — ARIA role-based search
3. `pw_code` — raw Playwright code parsed and executed

### Action String → Action Dict (used by agent)

```
"click [42]"                    → {action_type: CLICK, element_id: "42"}
"type [164] [query text] [1]"   → {action_type: TYPE, element_id: "164", text: [...], enter=True}
"scroll [down]"                 → {action_type: SCROLL, direction: "down"}
"stop [$279.49]"                → {action_type: STOP, answer: "$279.49"}
"goto [http://...]"             → {action_type: GOTO_URL, url: "..."}
"tab_focus [0]"                 → {action_type: PAGE_FOCUS, page_number: 0}
```

---

## Observation Processing (`processors.py`)

### TextObservationProcessor

Converts the live browser page into a text string the LLM can read.

**Pipeline:**

```
fetch_browser_info()
  └── CDP DOMSnapshot.captureSnapshot → raw DOM tree + viewport calibration

fetch_page_html() / fetch_page_accessibility_tree()
  └── CDP Accessibility.getFullAXTree → list of nodes with bounding boxes
  └── viewport filter: nodes with <60% visibility are dropped

parse_html() / parse_accessibility_tree()
  └── DFS traversal → flat string:
      "[42] button 'Submit'"
      "[43] textbox 'Search' focused: True"
  └── builds obs_nodes_info: {element_id → {backend_id, union_bound, text}}

get_element_center(element_id)
  └── looks up union_bound in obs_nodes_info
  └── returns normalized (x, y) in [0,1] for click targeting
```

**Why CDP instead of Playwright DOM API?**  
CDP gives more accurate bounding boxes and full accessibility tree in a single snapshot.

### ImageObservationProcessor

`page.screenshot()` → PNG bytes → numpy array `(height, width, 4)` RGBA.

### ObservationHandler

Orchestrates both processors. Returns `{"text": str, "image": ndarray}` each step.

### Key Data Structures

```python
DOMNode = {
    "nodeId": str,         # index in flattened tree (= element_id in actions)
    "nodeName": str,       # HTML tag
    "nodeValue": str,      # text content
    "attributes": str,     # HTML attributes string
    "backendNodeId": str,  # Chrome internal ID
    "parentId": str,
    "childIds": list[str],
    "union_bound": [x, y, w, h] | None,  # bounding box in viewport pixels
}

AccessibilityTreeNode = {
    "nodeId": str,
    "role": {"value": str},         # ARIA role (e.g. "button", "textbox")
    "name": {"value": str},         # accessible name
    "properties": list,             # ARIA properties
    "childIds": list[str],
    "union_bound": list[float] | None,
}
```

---

## Auto-Login (`auto_login.py`)

Before running a task, the browser needs to already be authenticated.

**`is_expired(storage_state, url, keyword)`**  
Loads saved cookies, navigates to the site, checks auth status by URL match or keyword in page text.

**`renew_comb(sites, auth_folder)`**  
Opens a browser, runs the login flow for each site (fills username/password), saves the resulting Playwright `storage_state` JSON (cookies + localStorage).

**Site-specific flows:**
- Shopping: `get_by_label` for email + password
- Reddit: `get_by_label` for username + password
- Shopping Admin: `get_by_placeholder` for username + password
- GitLab: `get_by_test_id` for username + password

At run time, `ScriptBrowserEnv.setup()` loads this JSON so the agent starts already logged in.

---

## Constants (`constants.py`)

| Constant | Value | Purpose |
|---|---|---|
| `UTTERANCE_MAX_LENGTH` | 8192 | max observation text length |
| `TYPING_MAX_LENGTH` | 64 | max chars per TYPE action |
| `MAX_ELEMENT_ID` | 1000 | max node IDs per page |
| `URL_MAX_LENGTH` | 256 | |
| `MAX_PAGE_NUMBER` | 10 | max open tabs |
| `ROLES` | 86 entries | standard ARIA roles |
| `ASCII_CHARSET` | chr 32–127 | printable ASCII for typing |
| `SPECIAL_KEYS` | Tab, Enter, ArrowUp, etc. | keyboard shortcut keys |
