import "ress/dist/ress.min.css"
import "./styles/global.css"

import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { App } from "./App"

const root = document.getElementById("root")

if (root === null) {
  throw new Error("Failed to initialize application.")
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>
)
