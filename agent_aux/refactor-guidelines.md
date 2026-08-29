# Refactor Guidelines (Agent-Oriented)

Concise rules for C++/CUDA refactors. Agent-readable, essential only.

---

## 0. Execution & Logging

| Rule | Action |
|------|--------|
| **Record all edits** | Log every file modified: path + brief change; maintain traceability. |
| **Command output truncation** | Build/run output large → inspect **last 100 lines only**; errors typically appear at the end. |

---

## 1. Access Control & Inheritance

| Rule | Action |
|------|--------|
| Composition > deep inheritance | Use interfaces + composition; avoid deep chains `A→B→C→D`. |
| Expose via interface, not protected | External callers use `I*`; no `protected` across translation units. |
| Audit before refactor | Grep `private:`/`protected:`; list methods that new code will call. |
| Avoid protected cross-unit | Need Foo→Bar across units? Add public wrapper in Bar; do not make Bar protected. |

---

## 2. Headers & Dependencies

| Rule | Action |
|------|--------|
| Forward declare | `class X;` in headers; `#include "X.h"` only in .cpp where full def needed. |
| Minimize includes | Each header includes only required deps. |
| Interfaces first | Declare `virtual` in interface; impls include interface, not vice versa. |
| Document forward decls | One-line comment per forward decl explaining purpose. |

---

## 3. Build & Validation

| Rule | Action |
|------|--------|
| Reduce CUDA rebuilds | Move non-CUDA logic to .cpp; keep .cu minimal; avoid touching widely-included headers. |
| Pimpl / interface split | Hide impl in .cpp/.cu to avoid header-change rebuilds. |
| Validate per phase | Run 1–2 scene regressions after each refactor step. |
| Cross-platform build | Prefer `cmake --build build`; PowerShell: use `;` not `&&`. |

---

## 4. Design & Process

| Rule | Action |
|------|--------|
| Design before code | Short design: interfaces, data flow, sync boundaries. |
| Incremental replacement | Add new wrappers; keep old path working until migration done. |
| Sync boundaries | Mark Host↔Device points with `[SYNC] UPLOAD/DOWNLOAD`; centralize in `agent_aux/`. |

---

## 5. Data Flow (Host/Device)

| Rule | Action |
|------|--------|
| CPU: host-only | No cudaMalloc/copy in CPU path. |
| GPU: device during step | Data on device in `step()`; sync only at init/reset/step boundaries. |
| Single source of truth | Explicit copy at boundaries; avoid duplicate Host/Device structures. |
| Annotate copies | One-line comment at each `copy_*2*`: purpose + direction. |

---

## 6. Checklist

**Before:** Audit target classes (public/protected/private); define interfaces; add forward decls; decide sync points.

**During:** Add wrappers; keep old path working; use `;` in PowerShell.

**After:** Build ok; run 1+ smoke test.

**Execution:** Log edits; for long output → last 100 lines.
