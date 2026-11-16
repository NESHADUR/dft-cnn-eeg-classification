# How to upload this project to GitHub

Everything in this folder is already packaged as a GitHub repository. Two routes below —
pick **Route B** if you can; it is faster, keeps history, and is what employers/instructors expect.

---

## 0. What is in the package

| File | Purpose | Upload? |
|---|---|:---:|
| `DFT_CNN_EEG.ipynb` | Main experiment notebook | ✅ |
| `Epileptic Seizure Recognition.csv` | Dataset (7.3 MB) | ✅ |
| `fig1…fig12 .png` | Result figures (~2.5 MB total) | ✅ |
| `model_dft_cnn.pth`, `model_raw_cnn.pth` | Trained weights (1.1 MB each) | ✅ |
| `README.md` | Renders on the repo homepage | ✅ |
| `requirements.txt` | Dependency install | ✅ |
| `LICENSE` | MIT licence | ✅ |
| `CITATION.cff` | "Cite this repository" button | ✅ |
| `.gitignore` | Keeps junk out | ✅ (hidden, but required) |
| `docs/` | Report (PDF/MD/DOCX) + extra documents | ✅ |
| `.ipynb_checkpoints/` | Jupyter autosaves | ❌ ignored |

All files are far below GitHub's 25 MB web-upload limit and 100 MB hard limit. No Git LFS needed.

---

## Route A — Upload through the GitHub website (no Git installed)

Good for a one-off upload. Limitation: you cannot update the repo afterwards without re-uploading.

1. **Sign in** at <https://github.com>.

2. **Create the repository**: click **+** (top right) → **New repository**.
   - *Repository name*: `dft-cnn-eeg-classification`
   - *Description*: `DFT-based frequency-domain preprocessing for 1D CNN EEG classification (PyTorch)`
   - **Public** (or Private, if your instructor allows)
   - **Do NOT** tick "Add a README file", "Add .gitignore" or "Choose a license" — you already have all three, and pre-created files cause a merge conflict on the first push.

3. On the empty repo page click **uploading an existing file** (or go to `https://github.com/<you>/<repo>/upload/main`).

4. **Drag and drop** everything from `D:\SEMESTER 06\Digital Signal Process` into the browser:
   - Drag the *files* (not the folder itself) — the notebook, CSV, the 12 PNGs, the two `.pth` files,
     `README.md`, `requirements.txt`, `LICENSE`, `CITATION.cff`, `.gitignore`.
   - ⚠️ **`.gitignore` is hidden in Explorer.** Enable *View → Show → Hidden items*, or the file will be
     silently skipped. Without it GitHub may offer to add its own later.
   - For the report files: in the upload page type `docs/` into the **"Name your file"** path field
     after dragging, or upload the `docs` folder by dragging the folder itself (Chrome/Edge support
     folder drag-and-drop; Firefox does not).

5. Scroll down, set the commit message to `Initial commit: DFT-CNN EEG classification project`,
   click **Commit changes**.

6. Verify: the repo homepage should render your README, with figures visible.

> If the web UI shows *"Yowza, that didn't work"*, you probably have >100 files or a file >25 MB —
> not the case here; retry in smaller batches.

---

## Route B — Push with Git (recommended)

Git 2.53 is already installed on this machine, and **I have already run `git init` and made the first
commit** in this folder. You only need to create the remote repo and push.

### Step 1 — Create an empty repository on GitHub

<https://github.com/new> → name it `dft-cnn-eeg-classification` → **Public** →
**do not** add README/gitignore/license → **Create repository**.

You will land on a page with commands like
`git remote add origin https://github.com/NESHADUR/dft-cnn-eeg-classification.git`. Keep that URL.

### Step 2 — Connect and push

Open **PowerShell** in this folder (or `cd` into it — note the quotes, the path has spaces):

```powershell
cd "D:\SEMESTER 06\Digital Signal Process"

# Point the local repo at GitHub (replace with YOUR username/repo)
git remote add origin https://github.com/NESHADUR/dft-cnn-eeg-classification.git

# Confirm which branch you are on (should be "main")
git branch -M main

# Push
git push -u origin main
```

### Step 3 — Authenticate

GitHub no longer accepts your account password over HTTPS. When a browser window pops up,
choose **"Sign in with your browser"** (Git Credential Manager) and approve. Alternatively paste a
**Personal Access Token**:

1. <https://github.com/settings/tokens> → *Generate new token (classic)*
2. Scope: ☑ `repo` → *Generate token* → copy it (`ghp_…`)
3. At the `Password` prompt, paste the token (nothing appears while typing — that is normal).

Success looks like:

```
Enumerating objects: 45, done.
Writing objects: 100% (45/45), 14.02 MiB | 3.41 MiB/s, done.
To https://github.com/NESHADUR/dft-cnn-eeg-classification.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
```

### Step 4 — Later updates

```powershell
git add -A
git commit -m "Describe what you changed"
git push
```

---

## Troubleshooting

| Error | Cause / fix |
|---|---|
| `src refspec main does not match any` | No commit exists yet: `git add -A; git commit -m "Initial commit"` |
| `remote origin already exists` | `git remote set-url origin <url>` (or `git remote remove origin` then re-add) |
| `Authentication failed` | Use a token, not your password. `gh auth login` also works. |
| `Updates were rejected because the remote contains work…` | You added a README on GitHub. `git pull --rebase origin main` then push. |
| `error: File … is 120.00 MB; this exceeds GitHub's file size limit` | Not applicable here (largest file is 7.3 MB). Otherwise use Git LFS. |
| `fatal: not a git repository` | You are in the wrong folder — `cd` into this one first. |
| Repo shows no README / figure broken | README must be at repo root; images must be uploaded alongside it (case-sensitive on GitHub). |
| Notebook renders as raw JSON | Normal for large notebooks; GitHub may fail to render >1 MB. Add the `.ipynb` preview via `nbviewer` link in the README, or install the GitHub "Rich Jupyter" extension. |

---

## After uploading — 5-minute polish

1. **About section** (top right of repo → ⚙): paste the description, add website if any.
2. **Topics**: `eeg` `dft` `cnn` `pytorch` `signal-processing` `deep-learning` `biomedical` `seizure-detection`.
3. **Releases** → *Draft a new release* → tag `v1.0.0` → attach the PDF report. Gives a citable version.
4. **Turn on Actions**? Not needed — there is no CI config, and that is fine for a course project.
5. **Update the placeholders** in `README.md` and `CITATION.cff`: replace `<your-username>/<your-repo>`
   and the hard-coded `NESHADUR` URL with your final URLs, then push again.
6. **Double-check nothing unintended is public**: `git ls-files` lists exactly what got committed.
   If a path has a space (the CSV does), its GitHub URL shows `%20` — harmless, but you may rename it
   to `data/epileptic_seizure_recognition.csv` and update the notebook's `DATA_PATH` if you prefer clean URLs.

### Optional cleanup: is every file appropriate?

`docs/DIP experiments-2025(AI).docx` and `docs/期末课程报告A-AI2026.pdf` belong to a different course
(DIP experiments / final course report). They do not break anything, but if you want a repository that is
purely about the DFT-CNN EEG study, delete them from `docs/` and adjust the structure listing in the README.
