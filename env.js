// Minimal .env loader — no dependencies (no dotenv needed).
// Usage:  const { MANTIS_API_KEY, MANTIS_API_BASE } = require("./env");
const fs = require("fs");
const path = require("path");

const vars = {};
const envPath = path.join(__dirname, ".env");
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
    if (m && !line.trim().startsWith("#")) {
      vars[m[1]] = m[2].replace(/^["']|["']$/g, "");
      if (!(m[1] in process.env)) process.env[m[1]] = vars[m[1]];
    }
  }
}

if (!vars.MANTIS_API_KEY && !process.env.MANTIS_API_KEY) {
  console.warn("warning: MANTIS_API_KEY is empty — add it to .env");
}

module.exports = {
  MANTIS_API_KEY: process.env.MANTIS_API_KEY || "",
  MANTIS_API_BASE:
    process.env.MANTIS_API_BASE || "https://kellis-h200-1.csail.mit.edu",
};
