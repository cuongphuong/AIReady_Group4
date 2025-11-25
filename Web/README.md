# BugClassifier Chat (Web)

BugClassifier Chat is a small Vite + React scaffold for a chat UI specialized in classifying bug reports. It lives in the `Web` folder.

Quick start (Windows Powershell):

```powershell
cd d:\AIReady_Group4\Web
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

What this includes:
- `src/components/ChatWindow.jsx` — messages list + composer
- `src/components/Message.jsx` — message bubble
- `@nextui-org/react` integrated via `NextUIProvider`

Next steps:
- Hook the composer to a real backend or OpenAI API.
- Add authentication, message streaming, and persistent history.
