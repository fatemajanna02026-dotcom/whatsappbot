require("dotenv").config();

const express   = require("express");
const http      = require("http");
const socketIo  = require("socket.io");
const { Client, LocalAuth } = require("whatsapp-web.js");
const QRCode    = require("qrcode");
const { OpenAI } = require("openai");
const path      = require("path");
const fs        = require("fs");

const app    = express();
const server = http.createServer(app);
const io     = socketIo(server, {
  cors: { origin: "*", methods: ["GET", "POST", "PATCH", "DELETE"] },
});

app.use(express.json({ limit: "2mb" }));
// public/ folder এ index.html থাকলে সেটা serve হবে
app.use(express.static(path.join(__dirname, "public")));

const PORT = Number(process.env.PORT || 3001);

// ══════════════════════════════════════════════════════════════
//  🔑 MULTI-API ROTATION SYSTEM
//  Priority: Groq (fastest/free) → OpenRouter → Cohere fallback
// ══════════════════════════════════════════════════════════════

// Groq keys (.env থেকে GROQ_API_KEY, GROQ_API_KEY1...12)
const GROQ_KEYS = [
  process.env.GROQ_API_KEY,
  ...Array.from({ length: 12 }, (_, i) => process.env[`GROQ_API_KEY${i + 1}`]),
].filter(Boolean).filter((v, i, a) => a.indexOf(v) === i);

// OpenRouter keys
const OPENROUTER_KEYS = [
  process.env.OPENROUTER_API_KEY,
  ...Array.from({ length: 6 }, (_, i) => process.env[`OPENROUTER_API_KEY${i + 1}`]),
].filter(Boolean).filter((v, i, a) => a.indexOf(v) === i);

// Cohere keys (last fallback)
const COHERE_KEYS = Array.from(
  { length: 12 }, (_, i) => process.env[`COHERE_KEY${i + 1}`]
).filter(Boolean);

// Per-key rate limit tracker: { key -> { failures, cooldownUntil, totalUsed } }
const keyState = new Map();
function getKeyState(key) {
  if (!keyState.has(key)) keyState.set(key, { failures: 0, cooldownUntil: 0, totalUsed: 0 });
  return keyState.get(key);
}
function markKeyFailed(key, cooldownMs = 65_000) {
  const s = getKeyState(key);
  s.failures++;
  // বারবার fail করলে বেশি cooldown
  const cd = s.failures >= 3 ? cooldownMs * 3 : cooldownMs;
  s.cooldownUntil = Date.now() + cd;
  log(`⚠️ Key cooldown: ...${key.slice(-8)} for ${cd / 1000}s (fail #${s.failures})`);
}
function markKeySuccess(key) {
  const s = getKeyState(key);
  s.failures = 0;
  s.totalUsed++;
}
function isKeyAvailable(key) {
  return Date.now() > getKeyState(key).cooldownUntil;
}

// Available keys থেকে least-used random pick
function pickKey(keys) {
  const available = keys.filter(isKeyAvailable);
  if (!available.length) {
    // সব cooldown এ? সবচেয়ে আগে ফিরবে সেটা দাও
    return [...keys].sort(
      (a, b) => getKeyState(a).cooldownUntil - getKeyState(b).cooldownUntil
    )[0] || null;
  }
  // least used গুলো থেকে random
  available.sort((a, b) => getKeyState(a).totalUsed - getKeyState(b).totalUsed);
  const pool = available.slice(0, Math.max(1, Math.ceil(available.length / 2)));
  return pool[Math.floor(Math.random() * pool.length)];
}

// Per-model failure tracker
const modelState = new Map();
function getModelState(m) {
  if (!modelState.has(m)) modelState.set(m, { failures: 0, cooldownUntil: 0 });
  return modelState.get(m);
}
function isModelAvailable(m) { return Date.now() > getModelState(m).cooldownUntil; }
function markModelFailed(m) {
  const s = getModelState(m);
  s.failures++;
  s.cooldownUntil = Date.now() + 120_000;
}

// Best free Groq models — quality order
const GROQ_MODELS = [
  "llama-3.3-70b-versatile",
  "llama-3.1-70b-versatile",
  "llama3-70b-8192",
  "mixtral-8x7b-32768",
  "gemma2-9b-it",
  "llama3-8b-8192",
];

// Best free OpenRouter models
const OPENROUTER_MODELS = [
  "meta-llama/llama-3.3-70b-instruct:free",
  "google/gemma-3-27b-it:free",
  "qwen/qwen-2.5-72b-instruct:free",
  "meta-llama/llama-3.1-8b-instruct:free",
  "mistralai/mistral-7b-instruct:free",
];

// ─── ডিরেক্টরি ────────────────────────────────────────────────
const DATA_DIR      = path.join(__dirname, "data");
const CONTACTS_DIR  = path.join(DATA_DIR, "contacts");
const NUMBERS_FILE  = path.join(DATA_DIR, "numbers.json");
const SETTINGS_FILE = path.join(DATA_DIR, "settings.json");

fs.mkdirSync(CONTACTS_DIR, { recursive: true });

// ─── লগিং ─────────────────────────────────────────────────────
function log(msg, data = "") {
  const ts = new Date().toLocaleTimeString();
  console.log(`[${ts}] ${msg}`, data);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function normalizeId(userId) {
  return String(userId || "").trim().replace(/[^\w-]/g, "_").slice(0, 80);
}

// ══════════════════════════════════════════════════════════════
//  ১. Settings
// ══════════════════════════════════════════════════════════════
const DEFAULT_SETTINGS = {
  autoReply         : true,
  ignoreGroups      : true,
  introMessage      : true,
  saveHistory       : true,
  aiEnabled         : true,
  notifyNewContact  : true,
  minDelayMs        : 900,
  maxDelayMs        : 6500,
  responseTemperature: 0.76,
};

function loadSettings() {
  try {
    if (fs.existsSync(SETTINGS_FILE)) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(fs.readFileSync(SETTINGS_FILE, "utf8")) };
    }
  } catch (_) {}
  return { ...DEFAULT_SETTINGS };
}

function saveSettings(patch) {
  const current = loadSettings();
  const updated = { ...current, ...patch };
  fs.writeFileSync(SETTINGS_FILE, JSON.stringify(updated, null, 2), "utf8");
  return updated;
}

// ══════════════════════════════════════════════════════════════
//  ২. Numbers Registry
// ══════════════════════════════════════════════════════════════
function loadNumbersRegistry() {
  try {
    if (fs.existsSync(NUMBERS_FILE)) {
      return JSON.parse(fs.readFileSync(NUMBERS_FILE, "utf8"));
    }
  } catch (_) {}
  return {};
}

function saveNumbersRegistry(registry) {
  fs.writeFileSync(NUMBERS_FILE, JSON.stringify(registry, null, 2), "utf8");
}

function registerNumber(phone, contactName, savedName) {
  const registry = loadNumbersRegistry();
  const now = new Date().toISOString();

  if (!registry[phone]) {
    registry[phone] = {
      phone,
      contactName : contactName || "Unknown",
      savedName   : savedName || "",
      firstSeen   : now,
      lastSeen    : now,
      msgCount    : 1,
    };
    log(`📋 নতুন নম্বর: ${phone} | ${contactName || savedName || "Unknown"}`);
  } else {
    registry[phone].lastSeen   = now;
    registry[phone].msgCount  += 1;
    if (contactName) registry[phone].contactName = contactName;
    if (savedName)   registry[phone].savedName   = savedName;
  }

  saveNumbersRegistry(registry);
  return registry[phone];
}

// ══════════════════════════════════════════════════════════════
//  ৩. Per-contact data
// ══════════════════════════════════════════════════════════════
function getContactFile(phone) {
  return path.join(CONTACTS_DIR, `${phone}.json`);
}

function loadContactData(phone) {
  const file = getContactFile(phone);
  try {
    if (fs.existsSync(file)) {
      return JSON.parse(fs.readFileSync(file, "utf8"));
    }
  } catch (_) {}

  return {
    phone,
    name      : "Unknown",
    savedName : "",
    memory    : {
      userPreferences : {},
      importantInfo   : [],
      lastFewTopics   : [],
      emotionalTone   : "neutral",
      repeatedQueries : {},
      saidBusy        : false,
      saidWhoIAm      : false,
      saidWillTell    : false,
      userName        : null,
      topic           : null,
      budget          : null,
      priority        : null,
    },
    history   : [],
    introSent : false,
    msgCount  : 0,
    lastSeen  : null,
    firstSeen : new Date().toISOString(),
  };
}

function saveContactData(phone, data) {
  data.lastSeen = new Date().toISOString();
  if (data.history.length > 50) {
    data.history = data.history.slice(-50);
  }
  fs.writeFileSync(getContactFile(phone), JSON.stringify(data, null, 2), "utf8");
}

// সব contact data একসাথে load (stats + sessions এর জন্য)
function loadAllContacts() {
  const registry = loadNumbersRegistry();
  const result   = {};

  // numbers.json থেকে base info
  for (const [phone, info] of Object.entries(registry)) {
    result[phone] = { ...info };
  }

  // contacts/*.json থেকে history + memory merge
  try {
    const files = fs.readdirSync(CONTACTS_DIR).filter(f => f.endsWith(".json"));
    for (const file of files) {
      const phone = file.replace(".json", "");
      try {
        const data = JSON.parse(fs.readFileSync(path.join(CONTACTS_DIR, file), "utf8"));
        result[phone] = {
          ...(result[phone] || {}),
          ...data,
          // registry থেকে নেওয়া msgCount priority দেব
          msgCount : (result[phone]?.msgCount) || data.msgCount || 0,
        };
      } catch (_) {}
    }
  } catch (_) {}

  return result;
}

// ══════════════════════════════════════════════════════════════
//  ৪. Sessions (connected WA clients)
// ══════════════════════════════════════════════════════════════
const clients = new Map();
// { userId -> { client, status, qrString, phone, connectedAt } }

function getSessionsList() {
  const list = [];
  for (const [userId, info] of clients.entries()) {
    list.push({
      id          : userId,
      userId,                        // index.html দুটোই use করে
      status      : info.status || "unknown",
      phone       : info.phone  || null,
      name        : info.name   || info.phone || userId,
      connectedAt : info.connectedAt || null,
    });
  }
  return list;
}

// ══════════════════════════════════════════════════════════════
//  ৫. Metrics
// ══════════════════════════════════════════════════════════════
function buildMetrics() {
  const allContacts = loadAllContacts();
  const phones      = Object.values(allContacts);
  const totalMsgs   = phones.reduce((s, p) => s + (p.msgCount || 0), 0);

  // আজকের মেসেজ
  const today = new Date().toDateString();
  const todayMsgs = phones.reduce((s, p) => {
    if (!p.lastSeen) return s;
    return new Date(p.lastSeen).toDateString() === today ? s + 1 : s;
  }, 0);

  // index.html renderMetrics() এ যেসব field দরকার সব দিচ্ছি
  return {
    totalContacts   : phones.length,
    totalMessages   : totalMsgs,
    todayMessages   : todayMsgs,
    activeSessions  : clients.size,
    repliesThisHour : activityLog.filter(a =>
      a.type === "reply" && Date.now() - new Date(a.time).getTime() < 3600000
    ).length,
    messagesIn      : totalMsgs,
    messagesOut     : activityLog.filter(a => a.type === "reply").length,
    aiStatus        : GROQ_KEYS.length > 0 ? "active" : "offline",
    avgResponseMs   : 0,
  };
}

// ══════════════════════════════════════════════════════════════
//  ৬. Activity Log (in-memory, max 200)
// ══════════════════════════════════════════════════════════════
const activityLog = [];

function pushActivity(type, message, phone = null) {
  const entry = {
    id      : Date.now(),
    type,   // 'message' | 'reply' | 'connect' | 'disconnect' | 'qr' | 'error'
    message,
    phone,
    time    : new Date().toISOString(),
  };
  activityLog.unshift(entry);
  if (activityLog.length > 200) activityLog.pop();
  io.emit("activity", entry);
  return entry;
}

// ══════════════════════════════════════════════════════════════
//  ৭. Dashboard State (index.html এর জন্য)
// ══════════════════════════════════════════════════════════════
function buildDashboardState() {
  const allContacts = loadAllContacts();
  const metrics     = buildMetrics();
  const sessions    = getSessionsList();

  return {
    profile  : {
      name     : "Sabbir",
      location : "Dhaka, Bangladesh",
      study    : "Programming & Technology",
    },
    // index.html renderSessions() এ session.userId, session.status, session.phone দরকার
    sessions : sessions.map(s => ({
      ...s,
      userId   : s.id,           // index.html 'userId' field use করে
      clientId : s.id,
    })),
    metrics,
    activity : activityLog.slice(0, 50),
    contacts : Object.values(allContacts).map(c => ({
      id          : c.phone,
      phone       : c.phone,
      displayName : c.savedName || c.contactName || c.name || c.phone,
      relation    : "auto",
      tone        : "auto",
      closeness   : 0,
      notes       : c.memory?.topic || "",
      msgCount    : c.msgCount || 0,
      lastSeen    : c.lastSeen,
      memory      : c.memory || {},
    })),
    settings : loadSettings(),
  };
}

// ══════════════════════════════════════════════════════════════
//  ৮. System Prompt
// ══════════════════════════════════════════════════════════════
function buildSystemPrompt(contactData) {
  const { memory, name, savedName, history } = contactData;
  const callerName = memory.userName || savedName || name || null;

  // শেষ ৫টা bot reply collect করো — এগুলো আর repeat করবে না
  const recentBotReplies = history
    .filter(h => h.role === "assistant")
    .slice(-5)
    .map(h => `"${h.content.substring(0, 60)}"`)
    .join("\n");

  // conversation এর context বোঝো
  const lastUserMsgs = history
    .filter(h => h.role === "user")
    .slice(-3)
    .map(h => h.content)
    .join(" | ");

  return `তুমি Sabbir ভাইয়ের personal AI Assistant। তোমার কাজ হলো তার WhatsApp-এ আসা মেসেজের বুদ্ধিমত্তার সাথে উত্তর দেওয়া।

## তোমার পরিচয়
- তুমি Sabbir ভাইয়ের AI Assistant — এটা শুধু প্রথমবার বলেছ, আর বলবে না
- Sabbir ভাই Dhaka-তে থাকেন, programming ও technology নিয়ে কাজ করেন, এখন ব্যস্ত
${callerName ? `- এই মানুষটার নাম/পরিচয়: ${callerName}` : ""}
${memory.topic ? `- এখন পর্যন্ত আলোচনার বিষয়: ${memory.topic}` : ""}
${memory.budget ? `- বাজেট উল্লেখ করেছেন: ${memory.budget}` : ""}
${memory.priority === "High" ? `- জরুরি বিষয় আছে` : ""}

## মেসেজ বোঝার নিয়ম — এটা সবচেয়ে গুরুত্বপূর্ণ
তুমি উত্তর দেওয়ার আগে মেসেজটা মনোযোগ দিয়ে পড়বে এবং বুঝবে:
- মানুষটা আসলে কী চাইছে? (কাজ, তথ্য, সাহায্য, নাকি শুধু কথা বলতে চাইছে)
- এটা কি আগের কথোপকথনের continuation?
- প্রশ্ন থাকলে সরাসরি সেই প্রশ্নের উত্তর দাও
- কাজের কথা থাকলে বলো Sabbir ভাই ফ্রি হলে জানাবেন
- সালাম বা শুভেচ্ছায় সংক্ষেপে সাড়া দাও

## উত্তর দেওয়ার নিয়ম
- উত্তর সর্বোচ্চ ২ লাইন, সরাসরি, কোনো ভূমিকা নেই
- ১০০% বাংলায় (tech/English শব্দ ছাড়া)
- "আপনি" ব্যবহার করো, ভদ্র থাকো
- context না বুঝলে একটা ছোট প্রশ্ন করো, assume করো না

## ⛔ এগুলো আগেই বলা হয়েছে — একদম বলবে না
- "আমি Sabbir ভাইয়ের Assistant" — intro দেওয়া হয়ে গেছে
- "Sabbir ভাই ব্যস্ত" — একবারের বেশি বলা যাবে না
- "জানিয়ে দেব / পৌঁছে দেব" — একবার বলার পর আর না
${recentBotReplies ? `\n## সাম্প্রতিক তোমার উত্তরগুলো (এগুলো repeat করবে না):\n${recentBotReplies}` : ""}

## উদাহরণ
User: "ভাই আছেন?"
✅ "জি আছি, বলুন।"
❌ "আমি Sabbir ভাইয়ের Assistant, উনি ব্যস্ত..."

User: "একটা website বানাতে চাই, কত নেবেন?"
✅ "ধন্যবাদ! বিস্তারিত বলুন, Sabbir ভাই ফ্রি হলে আলোচনা করবেন।"
❌ "Sabbir ভাই এখন ব্যস্ত আছেন, জানিয়ে দেব।"

User: "আপনাদের কি React জানা আছে?"
✅ "হ্যাঁ, React, Node.js সহ full-stack কাজ হয়।"
❌ "আমি তাকে পৌঁছে দেব।"`;
}

// ══════════════════════════════════════════════════════════════
//  ৯. Memory Updater
// ══════════════════════════════════════════════════════════════
function updateMemory(contactData, userMsg, botReply) {
  const msg = userMsg.toLowerCase();
  const rep = botReply.toLowerCase();
  const m   = contactData.memory;

  if (rep.includes("ব্যস্ত"))                                  m.saidBusy     = true;
  if (rep.includes("assistant") || rep.includes("পরিচয়"))      m.saidWhoIAm   = true;
  if (rep.includes("জানিয়ে দেব") || rep.includes("বলে দেব")) m.saidWillTell = true;
  if (rep.includes("জানাবো"))                                  m.saidWillTell = true;

  const nameMatch = msg.match(/(?:আমি|ami)\s+([^\s।,!?]+)/);
  if (nameMatch && !m.userName) m.userName = nameMatch[1];

  const budgetMatch = userMsg.match(/(\d+)\s*(taka|টাকা|tk)/i);
  if (budgetMatch) {
    m.budget = budgetMatch[0];
    m.topic  = "Budget Discussion";
  }

  if (userMsg.includes("জরুরি") || userMsg.includes("আজকে") || userMsg.includes("deadline")) {
    m.priority = "High";
  }

  const topics = {
    "কাজ": "কাজ", "টাকা": "টাকা", "পেমেন্ট": "পেমেন্ট",
    "project": "Project", "freelance": "Freelance",
    "ডিজাইন": "ডিজাইন", "ওয়েবসাইট": "ওয়েবসাইট",
    "সমস্যা": "সমস্যা", "জরুরি": "জরুরি বিষয়",
  };
  for (const [keyword, label] of Object.entries(topics)) {
    if (msg.includes(keyword)) { m.topic = label; break; }
  }
}

// ══════════════════════════════════════════════════════════════
//  ১০. AI Response — Multi-Provider Rotation
// ══════════════════════════════════════════════════════════════

// ── Groq দিয়ে চেষ্টা ─────────────────────────────────────────
async function tryGroq(systemPrompt, messages, temperature) {
  const availableModels = GROQ_MODELS.filter(isModelAvailable);
  if (!availableModels.length) throw new Error("All Groq models on cooldown");
  if (!GROQ_KEYS.length)       throw new Error("No Groq keys");

  const key = pickKey(GROQ_KEYS);
  if (!key) throw new Error("No Groq key available");

  const model = availableModels[0];

  const client = new OpenAI({
    baseURL : "https://api.groq.com/openai/v1",
    apiKey  : key,
    timeout : 18_000,
  });

  try {
    const res = await client.chat.completions.create({
      model,
      max_tokens  : 220,
      temperature,
      messages    : [{ role: "system", content: systemPrompt }, ...messages],
    });
    markKeySuccess(key);
    log(`✅ Groq OK: ${model} | key ...${key.slice(-6)}`);
    return res.choices?.[0]?.message?.content?.trim();
  } catch (err) {
    const msg = err.message || "";
    // Rate limit বা quota শেষ
    if (err.status === 429 || msg.includes("rate") || msg.includes("limit") || msg.includes("quota")) {
      markKeyFailed(key, 65_000);
      // model specific limit?
      if (msg.includes("model") || msg.includes("tokens_per_day")) markModelFailed(model);
    } else {
      markKeyFailed(key, 15_000);
    }
    throw err;
  }
}

// ── OpenRouter দিয়ে চেষ্টা ────────────────────────────────────
async function tryOpenRouter(systemPrompt, messages, temperature) {
  if (!OPENROUTER_KEYS.length) throw new Error("No OpenRouter keys");

  const key = pickKey(OPENROUTER_KEYS);
  if (!key) throw new Error("No OpenRouter key available");

  const availableModels = OPENROUTER_MODELS.filter(isModelAvailable);
  if (!availableModels.length) throw new Error("All OpenRouter models on cooldown");
  const model = availableModels[0];

  const client = new OpenAI({
    baseURL           : "https://openrouter.ai/api/v1",
    apiKey            : key,
    timeout           : 22_000,
    defaultHeaders    : {
      "HTTP-Referer"  : "https://github.com/sabbir-bot",
      "X-Title"       : "Sabbir WP Bot",
    },
  });

  try {
    const res = await client.chat.completions.create({
      model,
      max_tokens  : 220,
      temperature,
      messages    : [{ role: "system", content: systemPrompt }, ...messages],
    });
    markKeySuccess(key);
    log(`✅ OpenRouter OK: ${model} | key ...${key.slice(-6)}`);
    return res.choices?.[0]?.message?.content?.trim();
  } catch (err) {
    const msg = err.message || "";
    if (err.status === 429 || msg.includes("rate") || msg.includes("limit")) {
      markKeyFailed(key, 65_000);
      markModelFailed(model);
    } else {
      markKeyFailed(key, 15_000);
    }
    throw err;
  }
}

// ── Cohere fallback ────────────────────────────────────────────
async function tryCohere(systemPrompt, messages, temperature) {
  if (!COHERE_KEYS.length) throw new Error("No Cohere keys");

  const key = pickKey(COHERE_KEYS);
  if (!key) throw new Error("No Cohere key available");

  // Cohere REST API (openai-compatible endpoint নেই, তাই direct fetch)
  const chatHistory = messages.slice(0, -1).map(m => ({
    role    : m.role === "assistant" ? "CHATBOT" : "USER",
    message : m.content,
  }));
  const lastUser = messages.filter(m => m.role === "user").slice(-1)[0]?.content || "";

  const body = {
    model       : "command-r-plus",
    preamble    : systemPrompt,
    chat_history: chatHistory,
    message     : lastUser,
    temperature,
    max_tokens  : 200,
  };

  const res = await fetch("https://api.cohere.ai/v1/chat", {
    method  : "POST",
    headers : { "Authorization": `Bearer ${key}`, "Content-Type": "application/json" },
    body    : JSON.stringify(body),
    signal  : AbortSignal.timeout(20_000),
  });

  if (!res.ok) {
    markKeyFailed(key, res.status === 429 ? 65_000 : 15_000);
    throw new Error(`Cohere ${res.status}`);
  }

  const data = await res.json();
  markKeySuccess(key);
  log(`✅ Cohere OK | key ...${key.slice(-6)}`);
  return data.text?.trim();
}

// Key count log (server start এ দেখাবে)
function logKeyStatus() {
  log(`🔑 Groq: ${GROQ_KEYS.length} keys | OpenRouter: ${OPENROUTER_KEYS.length} keys | Cohere: ${COHERE_KEYS.length} keys`);
}

// ── reply পরিষ্কার করো ────────────────────────────────────────
function cleanReply(reply, contactData) {
  if (!reply || reply.length < 3) return null;

  reply = reply
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/^#{1,6}\s/gm, "")
    .replace(/as an ai|i'm an ai|language model|artificial intelligence/gi, "")
    .replace(/\n{2,}/g, "\n")
    .trim();

  // সর্বোচ্চ ২ লাইন
  reply = reply.split("\n").filter(l => l.trim()).slice(0, 2).join("\n");

  // repeat check — শেষ bot reply এর সাথে মিলে গেলে বদলাও
  const lastBot = contactData.history
    .filter(h => h.role === "assistant")
    .slice(-1)[0]?.content || "";

  if (lastBot && reply.substring(0, 30).toLowerCase() === lastBot.substring(0, 30).toLowerCase()) {
    return "জি বলুন, আর কিছু জানার আছে?";
  }

  return reply.substring(0, 350);
}

// ── Main: সব provider try করো ────────────────────────────────
async function getAiResponse(userMessage, contactData) {
  const settings   = loadSettings();
  const temperature = settings.responseTemperature || 0.72;

  const systemPrompt  = buildSystemPrompt(contactData);
  const recentHistory = contactData.history
    .slice(-12)
    .map(h => ({ role: h.role, content: h.content }));
  const messages = [...recentHistory, { role: "user", content: userMessage }];

  const providers = [
    { name: "Groq",        fn: () => tryGroq(systemPrompt, messages, temperature) },
    { name: "OpenRouter",  fn: () => tryOpenRouter(systemPrompt, messages, temperature) },
    { name: "Cohere",      fn: () => tryCohere(systemPrompt, messages, temperature) },
  ];

  for (const provider of providers) {
    try {
      const raw   = await provider.fn();
      const clean = cleanReply(raw, contactData);
      if (clean) {
        log(`🤖 [${provider.name}] replied`);
        return clean;
      }
    } catch (err) {
      log(`❌ ${provider.name} failed: ${err.message?.slice(0, 80)}`);
      // next provider তে যাও
    }
  }

  // সব provider fail করলে
  log("❌ All providers failed");
  return "একটু সমস্যা হচ্ছে, কিছুক্ষণ পরে আবার মেসেজ করুন।";
}

// Key status API তে দেখানোর জন্য
function getApiStatus() {
  const groqAvail = GROQ_KEYS.filter(isKeyAvailable).length;
  const orAvail   = OPENROUTER_KEYS.filter(isKeyAvailable).length;
  const coAvail   = COHERE_KEYS.filter(isKeyAvailable).length;
  return {
    groq        : { total: GROQ_KEYS.length,        available: groqAvail },
    openrouter  : { total: OPENROUTER_KEYS.length,  available: orAvail  },
    cohere      : { total: COHERE_KEYS.length,       available: coAvail  },
    models      : {
      groq       : GROQ_MODELS.map(m => ({ model: m, available: isModelAvailable(m) })),
      openrouter : OPENROUTER_MODELS.map(m => ({ model: m, available: isModelAvailable(m) })),
    },
  };
}
// ══════════════════════════════════════════════════════════════
//  ১১. WhatsApp Client
// ══════════════════════════════════════════════════════════════
function createClient(userId) {
  const cleanUserId = normalizeId(userId);

  // আগে থেকে connected থাকলে skip
  if (clients.has(userId) && clients.get(userId).status === "ready") {
    log(`⚠️ Client already ready: ${userId}`);
    return clients.get(userId).client;
  }

  const AUTH_DIR = path.join(__dirname, ".wwebjs_auth");

  const client = new Client({
    authStrategy : new LocalAuth({ clientId: cleanUserId, dataPath: AUTH_DIR }),
    puppeteer    : {
      headless : true,
      args     : ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
    },
  });

  const info = { client, status: "initializing", qrString: null, phone: null, connectedAt: null };
  clients.set(userId, info);

  // Session list push করো
  io.emit("sessions", getSessionsList());

  client.on("qr", async qr => {
    log(`📷 QR Generated: ${userId}`);
    info.status    = "qr_ready";
    info.qrString  = qr;
    const qrImage  = await QRCode.toDataURL(qr);
    io.emit("qr", { userId, qrImage });
    io.emit("status", { userId, status: "qr_ready" });
    io.emit("sessions", getSessionsList());
    pushActivity("qr", `QR ready for ${userId}`);
  });

  client.on("ready", async () => {
    log(`✅ WhatsApp Connected: ${userId}`);
    info.status      = "ready";
    info.connectedAt = new Date().toISOString();

    // phone number — ready event এর পরে client.info available হয়
    try {
      await sleep(500); // একটু wait করো info populate হতে
      const wid    = client.info?.wid;
      const me     = client.info?.pushname || "";
      info.phone   = wid ? wid.user : null;
      info.name    = me;
      log(`📱 Phone: ${info.phone} | Name: ${info.name}`);
    } catch (_) {}

    io.emit("status", {
      userId,
      status : "ready",
      phone  : info.phone,
      name   : info.name,
    });
    io.emit("sessions",       getSessionsList());
    io.emit("dashboard-state", buildDashboardState());
    io.emit("metrics",         buildMetrics());
    pushActivity("connect", `${userId} connected — ${info.phone || "unknown"}`);
  });

  client.on("disconnected", reason => {
    log(`🔌 Disconnected: ${userId} — ${reason}`);
    info.status = "disconnected";
    io.emit("status", { userId, status: "disconnected", reason });
    io.emit("sessions", getSessionsList());
    pushActivity("disconnect", `${userId} disconnected: ${reason}`);
  });

  client.on("auth_failure", () => {
    log(`❌ Auth Failure: ${userId}`);
    info.status = "auth_failure";
    io.emit("status", { userId, status: "auth_failure" });
    io.emit("sessions", getSessionsList());
    pushActivity("error", `Auth failure for ${userId}`);
  });

  client.on("message", async msg => {
    if (msg.fromMe)                return;
    if (msg.from.includes("@g.us")) return;

    const settings = loadSettings();
    if (!settings.autoReply) return;

    const phone = msg.from.split("@")[0];
    const text  = msg.body?.trim() || "";
    if (!text) return;

    let contactName = "";
    let savedName   = "";
    try {
      const contact = await msg.getContact();
      contactName   = contact.pushname || "";
      savedName     = contact.name    || "";
    } catch (_) {}

    log(`📨 [${phone}] ${savedName || contactName || "Unknown"}: ${text.substring(0, 60)}`);

    const contactData    = loadContactData(phone);
    contactData.name     = contactName || contactData.name;
    contactData.savedName = savedName  || contactData.savedName;
    contactData.msgCount = (contactData.msgCount || 0) + 1;

    registerNumber(phone, contactName, savedName);

    // Live update সব socket-এ
    const livePayload = {
      phone,
      name    : savedName || contactName || phone,
      message : text.substring(0, 100),
      time    : new Date().toISOString(),
    };
    io.emit("new_message", livePayload);
    pushActivity("message", `From ${livePayload.name}: ${text.substring(0, 60)}`, phone);
    io.emit("metrics", buildMetrics());

    try {
      // ── ইন্ট্রো (শুধু প্রথমবার) ─────────────────────────────
      if (!contactData.introSent && settings.introMessage) {
        const callerName = savedName || contactName;
        const greeting   = callerName ? `আসসালামু আলাইকুম ${callerName} ভাই 😊` : "আসসালামু আলাইকুম 😊";

        const minD = settings.minDelayMs || 900;
        const maxD = settings.maxDelayMs || 6500;

        await sleep(minD);
        await msg.reply(`${greeting} আমি Sabbir ভাইয়ের AI Assistant বলছি।\nSabbir ভাই এখন coding নিয়ে ব্যস্ত আছেন।`);
        await sleep(Math.min(2500, maxD));
        await msg.reply("জরুরি কিছু বলার থাকলে আমাকে জানান, আমি তাকে পৌঁছে দেব। 🙏");
        await sleep(2000);
        await msg.reply("আপনি কি কিছু বলতে চাইছেন?");

        contactData.introSent = true;
        contactData.history.push({
          role    : "assistant",
          content : `${greeting} আমি Sabbir ভাইয়ের AI Assistant। উনি ব্যস্ত, জরুরি কিছু থাকলে বলুন।`,
          time    : new Date().toISOString(),
        });

        saveContactData(phone, contactData);
        pushActivity("reply", `Intro sent to ${livePayload.name}`, phone);
        return;
      }

      // ── সাধারণ কথোপকথন ──────────────────────────────────────
      if (!settings.aiEnabled) return;

      const chat = await msg.getChat();
      await chat.sendStateTyping();

      const minD = settings.minDelayMs || 900;
      await sleep(minD + Math.random() * 800);

      contactData.history.push({
        role    : "user",
        content : text,
        time    : new Date().toISOString(),
      });

      const aiReply = await getAiResponse(text, contactData);
      await msg.reply(aiReply);

      contactData.history.push({
        role    : "assistant",
        content : aiReply,
        time    : new Date().toISOString(),
      });

      updateMemory(contactData, text, aiReply);
      saveContactData(phone, contactData);
      pushActivity("reply", `Reply to ${livePayload.name}: ${aiReply.substring(0, 60)}`, phone);

      log(`✉️  [${phone}] Reply sent (#${contactData.msgCount})`);
    } catch (err) {
      log("❌ Message Error:", err.message);
      pushActivity("error", `Error handling message from ${phone}: ${err.message}`);
      try { await msg.reply("দুঃখিত, একটু সমস্যা হচ্ছে।"); } catch (_) {}
    }
  });

  client.initialize();
  return client;
}

// ══════════════════════════════════════════════════════════════
//  ১২. API Routes
// ══════════════════════════════════════════════════════════════

// ── Dashboard full state (index.html এর জন্য) ─────────────────
app.get("/api/dashboard-state", (_req, res) => {
  res.json(buildDashboardState());
});

// ── Sessions list ─────────────────────────────────────────────
app.get("/api/sessions", (_req, res) => {
  res.json(getSessionsList());
});

// ── Session action: pause / remove / reconnect ────────────────
app.post("/api/sessions/:action", async (req, res) => {
  const { action } = req.params;
  const userId = req.body?.userId || req.query?.userId;

  if (!userId) return res.status(400).json({ success: false, error: "userId required" });

  if (action === "pause_user") {
    const info = clients.get(userId);
    if (info) {
      try {
        await info.client.destroy();
      } catch (_) {}
      info.status = "paused";
      io.emit("status", { userId, status: "paused" });
      io.emit("sessions", getSessionsList());
      pushActivity("disconnect", `${userId} paused`);
    }
    return res.json({ success: true });
  }

  if (action === "remove_user") {
    const info = clients.get(userId);
    if (info) {
      try { await info.client.destroy(); } catch (_) {}
      // session auth folder মুছে দাও
      const authPath = path.join(__dirname, ".wwebjs_auth", `session-${normalizeId(userId)}`);
      try { fs.rmSync(authPath, { recursive: true, force: true }); } catch (_) {}
      clients.delete(userId);
      io.emit("status", { userId, status: "removed" });
      io.emit("sessions", getSessionsList());
      pushActivity("disconnect", `${userId} removed`);
    }
    return res.json({ success: true });
  }

  if (action === "reconnect_user") {
    const info = clients.get(userId);
    if (info) {
      try { await info.client.destroy(); } catch (_) {}
      clients.delete(userId);
    }
    createClient(userId);
    return res.json({ success: true });
  }

  res.status(400).json({ success: false, error: "Unknown action" });
});

// ── Connect (HTTP) ─────────────────────────────────────────────
app.post("/api/connect", (req, res) => {
  const userId = req.body?.userId || "sabbir";
  try {
    createClient(userId);
    res.json({ success: true, userId });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

// ── Settings ──────────────────────────────────────────────────
app.get("/api/settings", (_req, res) => {
  res.json(loadSettings());
});

app.patch("/api/settings", (req, res) => {
  try {
    const updated = saveSettings(req.body || {});
    io.emit("settings", updated);
    res.json({ success: true, settings: updated });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// ── Stats ─────────────────────────────────────────────────────
app.get("/api/stats", (_req, res) => {
  const allContacts = loadAllContacts();
  const phones      = Object.values(allContacts);
  res.json({
    totalNumbers  : phones.length,
    totalMessages : phones.reduce((s, p) => s + (p.msgCount || 0), 0),
    numbers       : allContacts,
  });
});

// ── All numbers ───────────────────────────────────────────────
app.get("/api/numbers", (_req, res) => {
  res.json(loadNumbersRegistry());
});

// ── Single chat history ───────────────────────────────────────
app.get("/api/chat/:phone", (req, res) => {
  const phone = req.params.phone;
  const data  = loadContactData(phone);
  res.json(data);
});

// ── Contacts CRUD ──────────────────────────────────────────────
app.get("/api/contacts", (_req, res) => {
  const allContacts = loadAllContacts();
  const list = Object.values(allContacts).map(c => ({
    id          : c.phone,
    phone       : c.phone,
    displayName : c.savedName || c.contactName || c.name || c.phone,
    relation    : "auto",
    tone        : "auto",
    closeness   : 0,
    notes       : c.memory?.topic || "",
    msgCount    : c.msgCount || 0,
    lastSeen    : c.lastSeen,
    memory      : c.memory || {},
  }));
  res.json(list);
});

app.put("/api/contacts/:id", (req, res) => {
  const phone    = req.params.id;
  const patch    = req.body || {};
  const existing = loadContactData(phone);

  if (patch.displayName) existing.savedName = patch.displayName;
  if (patch.notes !== undefined) {
    existing.memory = existing.memory || {};
    existing.memory.topic = patch.notes;
  }

  saveContactData(phone, existing);

  // numbers.json ও update করো
  const registry = loadNumbersRegistry();
  if (registry[phone]) {
    if (patch.displayName) registry[phone].savedName = patch.displayName;
    saveNumbersRegistry(registry);
  }

  res.json({ success: true });
});

app.delete("/api/contacts/:id", (req, res) => {
  const phone   = req.params.id;
  const file    = getContactFile(phone);
  const registry = loadNumbersRegistry();

  try { if (fs.existsSync(file)) fs.unlinkSync(file); } catch (_) {}
  if (registry[phone]) {
    delete registry[phone];
    saveNumbersRegistry(registry);
  }

  pushActivity("message", `Contact ${phone} deleted`);
  res.json({ success: true });
});

// ── Test reply ────────────────────────────────────────────────
app.post("/api/test-reply", async (req, res) => {
  const { senderId, message } = req.body || {};
  if (!message) return res.status(400).json({ error: "message required" });

  const contactData = loadContactData(senderId || "test_user");
  try {
    const reply = await getAiResponse(message, contactData);
    res.json({ reply });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── API Key Status ────────────────────────────────────────────
app.get("/api/api-status", (_req, res) => {
  res.json(getApiStatus());
});

// ── Activity log ──────────────────────────────────────────────
app.get("/api/activity", (_req, res) => {
  res.json(activityLog.slice(0, 100));
});

// ══════════════════════════════════════════════════════════════
//  ১৩. Web Panel (built-in simple panel)
// ══════════════════════════════════════════════════════════════
app.get("/panel", (_req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="bn">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sabbir Bot — Control Panel</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;display:flex;min-height:100vh}
    .sidebar{width:300px;background:#1a1d27;border-right:1px solid #2d3748;display:flex;flex-direction:column;flex-shrink:0}
    .sidebar-header{padding:20px;background:#667eea;color:white;font-weight:bold;font-size:18px}
    .sidebar-header small{display:block;font-size:12px;opacity:.8;margin-top:2px}
    .contact-list{overflow-y:auto;flex:1}
    .contact-item{padding:14px 18px;border-bottom:1px solid #2d3748;cursor:pointer;transition:.2s}
    .contact-item:hover{background:#2d3748}
    .contact-item.active{background:#2d3748;border-left:3px solid #667eea}
    .contact-name{font-weight:600;font-size:14px;color:#e2e8f0}
    .contact-phone{font-size:11px;color:#718096;margin-top:2px}
    .contact-last{font-size:11px;color:#4a5568;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .contact-badge{background:#667eea;color:white;border-radius:10px;padding:1px 7px;font-size:10px;float:right}
    .main{flex:1;display:flex;flex-direction:column}
    .topbar{padding:15px 25px;background:#1a1d27;border-bottom:1px solid #2d3748;display:flex;align-items:center;gap:15px;flex-wrap:wrap}
    .status-dot{width:10px;height:10px;border-radius:50%;background:#e53e3e;flex-shrink:0}
    .status-dot.connected{background:#38a169}
    #status-text{font-size:14px;color:#a0aec0}
    .btn{padding:8px 18px;border:none;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;transition:.2s}
    .btn-primary{background:#667eea;color:white;margin-left:auto}
    .btn-primary:hover{background:#5a67d8}
    .btn-danger{background:#e53e3e;color:white}
    .btn-danger:hover{background:#c53030}
    .btn-warn{background:#d69e2e;color:white}
    .btn-warn:hover{background:#b7791f}
    .chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
    .chat-header{padding:15px 25px;background:#1e2130;border-bottom:1px solid #2d3748;display:flex;align-items:center;justify-content:space-between}
    .chat-header h2{font-size:16px}
    .chat-header p{font-size:12px;color:#718096}
    .chat-actions{display:flex;gap:8px}
    .messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:10px}
    .message{max-width:70%;padding:10px 15px;border-radius:12px;font-size:14px;line-height:1.5}
    .message.user{background:#2d3748;align-self:flex-start;border-radius:12px 12px 12px 3px}
    .message.bot{background:#667eea;align-self:flex-end;border-radius:12px 12px 3px 12px}
    .message .time{font-size:10px;opacity:.6;margin-top:4px;text-align:right}
    #qr-section{display:flex;align-items:center;justify-content:center;flex:1;flex-direction:column}
    #qr-section img{max-width:250px;border-radius:12px;border:3px solid #667eea}
    .qr-text{text-align:center;color:#718096;font-size:13px;margin-top:12px}
    .live-feed{padding:10px 15px;background:#1a1d27;border-top:1px solid #2d3748;font-size:12px;color:#718096;min-height:36px}
    .live-feed span{color:#38a169}
    .empty{flex:1;display:flex;align-items:center;justify-content:center;color:#4a5568;font-size:14px}
    .memory-panel{padding:12px 20px;background:#1e2130;border-top:1px solid #2d3748;font-size:12px;color:#718096;max-height:80px;overflow:hidden}
    .memory-panel span{color:#a0aec0}
  </style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-header">
    🤖 Sabbir Bot
    <small id="total-label">লোড হচ্ছে...</small>
  </div>
  <div class="contact-list" id="contact-list">
    <div style="padding:20px;color:#4a5568;font-size:13px">কোনো কথোপকথন নেই</div>
  </div>
</div>

<div class="main">
  <div class="topbar">
    <div class="status-dot" id="status-dot"></div>
    <span id="status-text">WhatsApp সংযুক্ত নয়</span>
    <button class="btn btn-primary" id="connectBtn" onclick="connectBot()">🔗 Connect</button>
    <button class="btn btn-warn" onclick="reconnectBot()" style="display:none" id="reconnectBtn">🔄 Reconnect</button>
    <button class="btn btn-danger" onclick="removeBot()" style="display:none" id="removeBtn">🗑 Remove</button>
  </div>

  <div id="qr-section">
    <div style="text-align:center">
      <div id="qr-container"></div>
      <p class="qr-text">Connect বাটন চাপুন, তারপর WhatsApp দিয়ে QR স্ক্যান করুন</p>
    </div>
  </div>

  <div class="chat-area" id="chat-area" style="display:none">
    <div class="chat-header">
      <div>
        <h2 id="chat-name">—</h2>
        <p id="chat-phone">—</p>
      </div>
      <div class="chat-actions">
        <button class="btn btn-danger" onclick="deleteContact()" style="font-size:11px;padding:5px 10px">🗑 Delete</button>
      </div>
    </div>
    <div class="messages" id="messages"></div>
    <div class="memory-panel" id="memory-panel"></div>
  </div>

  <div class="live-feed" id="live-feed">⚡ Live: অপেক্ষায় আছি...</div>
</div>

<script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
<script>
const socket = io();
let activePhone = null;
let currentUserId = "sabbir";
let isConnected = false;

socket.on("status", data => {
  if (data.status === "ready") {
    isConnected = true;
    document.getElementById("status-dot").classList.add("connected");
    document.getElementById("status-text").textContent = "✅ WhatsApp Connected — " + (data.phone || "");
    document.getElementById("qr-section").style.display = "none";
    document.getElementById("connectBtn").style.display = "none";
    document.getElementById("reconnectBtn").style.display = "inline-block";
    document.getElementById("removeBtn").style.display = "inline-block";
  } else if (data.status === "disconnected" || data.status === "removed" || data.status === "paused") {
    isConnected = false;
    document.getElementById("status-dot").classList.remove("connected");
    document.getElementById("status-text").textContent = "❌ সংযুক্ত নয়";
    document.getElementById("connectBtn").style.display = "inline-block";
    document.getElementById("reconnectBtn").style.display = "none";
    document.getElementById("removeBtn").style.display = "none";
    if (data.status === "removed") document.getElementById("qr-section").style.display = "flex";
  } else if (data.status === "qr_ready") {
    document.getElementById("status-text").textContent = "📷 QR স্ক্যান করুন...";
  }
});

socket.on("qr", data => {
  document.getElementById("qr-container").innerHTML = '<img src="' + data.qrImage + '">';
  document.getElementById("qr-section").style.display = "flex";
  document.getElementById("chat-area").style.display = "none";
});

socket.on("new_message", data => {
  const name = data.name || data.phone;
  document.getElementById("live-feed").innerHTML = '⚡ নতুন মেসেজ — <span>' + name + '</span>: ' + escHtml(data.message);
  loadContacts();
  if (activePhone === data.phone) openChat(data.phone);
});

socket.on("sessions", () => { /* sessions updated */ });

function connectBot() {
  socket.emit("connect_bot", currentUserId);
  document.getElementById("status-text").textContent = "সংযোগ হচ্ছে...";
}

function reconnectBot() {
  fetch("/api/sessions/reconnect_user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userId: currentUserId }),
  }).then(() => {
    document.getElementById("status-text").textContent = "Reconnecting...";
    document.getElementById("reconnectBtn").style.display = "none";
    document.getElementById("removeBtn").style.display = "none";
    document.getElementById("connectBtn").style.display = "inline-block";
  });
}

function removeBot() {
  if (!confirm("WhatsApp session সম্পূর্ণ remove করবেন?")) return;
  fetch("/api/sessions/remove_user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userId: currentUserId }),
  });
}

function deleteContact() {
  if (!activePhone) return;
  if (!confirm(activePhone + " এর সব ডেটা ডিলিট করবেন?")) return;
  fetch("/api/contacts/" + activePhone, { method: "DELETE" })
    .then(() => { activePhone = null; loadContacts(); document.getElementById("chat-area").style.display = "none"; document.getElementById("qr-section").style.display = "flex"; });
}

function escHtml(s) {
  return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

async function loadContacts() {
  try {
    const res     = await fetch("/api/stats");
    const data    = await res.json();
    const numbers = Object.values(data.numbers || {});
    const list    = document.getElementById("contact-list");
    const label   = document.getElementById("total-label");

    label.textContent = numbers.length + " জন | " + data.totalMessages + " মেসেজ";

    if (!numbers.length) {
      list.innerHTML = '<div style="padding:20px;color:#4a5568;font-size:13px">কোনো মেসেজ আসেনি</div>';
      return;
    }

    numbers.sort((a, b) => new Date(b.lastSeen || 0) - new Date(a.lastSeen || 0));

    list.innerHTML = numbers.map(n => {
      const displayName = n.savedName || n.contactName || n.name || n.phone;
      const time = n.lastSeen ? new Date(n.lastSeen).toLocaleTimeString("bn-BD", {hour:"2-digit",minute:"2-digit"}) : "";
      return '<div class="contact-item ' + (activePhone===n.phone?"active":"") + '" onclick="openChat(\'' + n.phone + '\')">' +
        '<span class="contact-badge">' + (n.msgCount||0) + '</span>' +
        '<div class="contact-name">' + escHtml(displayName) + '</div>' +
        '<div class="contact-phone">' + escHtml(n.phone) + '</div>' +
        '<div class="contact-last">' + escHtml(time) + '</div>' +
        '</div>';
    }).join("");
  } catch(e) {}
}

async function openChat(phone) {
  activePhone = phone;
  document.getElementById("qr-section").style.display = "none";
  document.getElementById("chat-area").style.display = "flex";

  try {
    const res  = await fetch("/api/chat/" + phone);
    const data = await res.json();

    const displayName = data.savedName || data.name || phone;
    document.getElementById("chat-name").textContent  = displayName;
    document.getElementById("chat-phone").textContent = phone;

    // memory panel
    const m = data.memory || {};
    const memParts = [];
    if (m.userName)  memParts.push("নাম: " + m.userName);
    if (m.topic)     memParts.push("বিষয়: " + m.topic);
    if (m.budget)    memParts.push("বাজেট: " + m.budget);
    if (m.priority)  memParts.push("Priority: " + m.priority);
    document.getElementById("memory-panel").innerHTML = memParts.length
      ? "🧠 Memory: <span>" + memParts.join(" · ") + "</span>"
      : "";

    const msgDiv = document.getElementById("messages");
    if (!data.history || !data.history.length) {
      msgDiv.innerHTML = '<div class="empty">কোনো মেসেজ নেই</div>';
    } else {
      msgDiv.innerHTML = data.history.map(h => {
        const t = h.time ? new Date(h.time).toLocaleTimeString("bn-BD",{hour:"2-digit",minute:"2-digit"}) : "";
        return '<div class="message ' + (h.role==="user"?"user":"bot") + '">' +
          escHtml(h.content) + '<div class="time">' + escHtml(t) + '</div></div>';
      }).join("");
      msgDiv.scrollTop = msgDiv.scrollHeight;
    }
  } catch(e) {}

  loadContacts();
}

loadContacts();
setInterval(loadContacts, 15000);
</script>
</body>
</html>`);
});

// ══════════════════════════════════════════════════════════════
//  ১৪. Socket Events
// ══════════════════════════════════════════════════════════════
io.on("connection", socket => {
  log("Socket connected:", socket.id);

  // সংযোগ হলেই current state পাঠাও
  socket.emit("dashboard-state", buildDashboardState());
  socket.emit("sessions", getSessionsList());

  socket.on("connect_bot", userId => {
    try {
      createClient(userId || "sabbir");
      socket.emit("status", { userId: userId || "sabbir", status: "initializing" });
    } catch (err) {
      socket.emit("error", { message: err.message });
    }
  });

  socket.on("pause_user", userId => {
    const info = clients.get(userId);
    if (info) {
      info.client.destroy().catch(() => {});
      info.status = "paused";
      io.emit("status", { userId, status: "paused" });
      io.emit("sessions", getSessionsList());
    }
  });

  socket.on("remove_user", userId => {
    const info = clients.get(userId);
    if (info) {
      info.client.destroy().catch(() => {});
      const authPath = path.join(__dirname, ".wwebjs_auth", `session-${normalizeId(userId)}`);
      try { fs.rmSync(authPath, { recursive: true, force: true }); } catch (_) {}
      clients.delete(userId);
      io.emit("status", { userId, status: "removed" });
      io.emit("sessions", getSessionsList());
      pushActivity("disconnect", `${userId} removed`);
    }
  });

  socket.on("reconnect_user", userId => {
    const info = clients.get(userId);
    if (info) {
      info.client.destroy().catch(() => {});
      clients.delete(userId);
    }
    createClient(userId || "sabbir");
  });

  socket.on("disconnect", () => log("Socket disconnected:", socket.id));
});

// ══════════════════════════════════════════════════════════════
//  ১৫. Server Start
// ══════════════════════════════════════════════════════════════
server.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════╗
║         🤖 SABBIR BOT STARTED               ║
╠══════════════════════════════════════════════╣
║  🌐 Panel rr : http://localhost:${PORT}/panel      ║
║  🌐 Neural : http://localhost:${PORT}/           ║
║  🔑 Groq   : ${GROQ_KEYS.length} keys | OR: ${OPENROUTER_KEYS.length} | Co: ${COHERE_KEYS.length}  ║
║  💾 Data   : ./data/                         ║
╚══════════════════════════════════════════════╝
  `);
  logKeyStatus();
  if (!GROQ_KEYS.length && !OPENROUTER_KEYS.length) {
    console.warn("⚠️  কোনো API key নেই! .env ফাইলে GROQ_API_KEY বা OPENROUTER_API_KEY দিন।");
  }
});
