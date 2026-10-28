// DroneGen Studio: projects and chats on the left, the dialog in the middle, the figure on the right.
// Plain ES module, no build step; talks to the local server (/studio/api, /files, /blender).

const $ = (sel, root = document) => root.querySelector(sel);
const TOKEN = new URLSearchParams(location.search).get("token") || sessionStorage.getItem("dg-token") || "";
if (TOKEN) sessionStorage.setItem("dg-token", TOKEN);

const state = {
  projects: [], chats: [], chatId: null, chat: null, busy: new Set(), pending: null,
  es: null, ges: null, attachments: [], status: null, current: null, cards: new Map(), lastMesh: null,
};

// -- small helpers -----------------------------------------------------------------------
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) n.append(kid.nodeType ? kid : String(kid));
  return n;
}
const icon = (name) => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("class", "i");
  const u = document.createElementNS("http://www.w3.org/2000/svg", "use");
  u.setAttribute("href", `#i-${name}`);
  s.append(u);
  return s;
};
const withToken = (url) => (TOKEN ? url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(TOKEN) : url);
const fileUrl = (name) => withToken(`/files/${encodeURIComponent(name)}`);
const chatImageUrl = (name) => withToken(`/studio/api/images/${encodeURIComponent(name)}`);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function api(method, path, body, isForm = false) {
  const opts = { method, headers: {} };
  if (TOKEN) opts.headers.Authorization = `Bearer ${TOKEN}`;
  if (body !== undefined) {
    if (isForm) opts.body = body;
    else { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  }
  let r;
  try { r = await fetch(path, opts); } catch { throw new Error("Сервер DroneGen не отвечает — он запущен?"); }
  const data = r.headers.get("content-type")?.includes("json") ? await r.json() : null;
  if (!r.ok) throw new Error(data?.detail || `Ошибка ${r.status}`);
  return data;
}

function toast(text, kind = "") {
  const t = el("div", { class: `toast ${kind}` }, text);
  $("#toasts").append(t);
  setTimeout(() => t.remove(), kind === "error" ? 7000 : 3500);
}
async function guard(fn) {
  try { return await fn(); } catch (e) { toast(e.message, "error"); }
}

// A tiny, safe Markdown: paragraphs, lists, **bold**, *italic*, `code`, links.
function md(text) {
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const out = [];
  let list = null;
  for (const line of String(text || "").split(/\n/)) {
    const m = line.match(/^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$/);
    if (m) {
      const kind = m[1] ? "ol" : "ul";
      if (!list || list.kind !== kind) { if (list) out.push(`</${list.kind}>`); out.push(`<${kind}>`); list = { kind }; }
      out.push(`<li>${inline(m[2])}</li>`);
      continue;
    }
    if (list) { out.push(`</${list.kind}>`); list = null; }
    if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push(`</${list.kind}>`);
  return out.join("");
}

function openLightbox(src) {
  const lb = $("#lightbox");
  $("img", lb).src = src;
  lb.classList.add("on");
}
$("#lightbox").addEventListener("click", () => $("#lightbox").classList.remove("on"));

// -- sidebar -----------------------------------------------------------------------------
async function loadSidebar() {
  const q = $("#search").value.trim();
  [state.projects, state.chats] = await Promise.all([
    api("GET", "/studio/api/projects"),
    api("GET", "/studio/api/chats" + (q ? `?q=${encodeURIComponent(q)}` : "")),
  ]);
  for (const c of state.chats) if (c.busy) state.busy.add(c.id);
  renderSidebar();
}

function renderSidebar() {
  const nav = $("#projects");
  nav.replaceChildren();
  const q = $("#search").value.trim();
  for (const p of state.projects) {
    const chats = state.chats.filter((c) => c.project_id === p.id);
    if (q && !chats.length) continue;
    const head = el("div", { class: "project-head", title: "Проект (шоу)" },
      p.pinned ? icon("pin") : null,
      el("span", { class: "name" }, p.title),
      el("span", { class: "tools" },
        el("button", { class: "icon-btn", title: "Новый чат в этом проекте", onclick: (e) => { e.stopPropagation(); newChat(p.id); } }, icon("plus")),
        el("button", { class: "icon-btn", title: "Переименовать", onclick: (e) => { e.stopPropagation(); renameProject(p); } }, icon("edit")),
        el("button", { class: "icon-btn", title: "Удалить проект со всеми чатами", onclick: (e) => { e.stopPropagation(); deleteProject(p); } }, icon("trash"))));
    const box = el("div", { class: "project" }, head);
    for (const c of chats) {
      const item = el("div", { class: "chat-item" + (c.id === state.chatId ? " active" : ""), onclick: () => openChat(c.id) },
        c.pinned ? el("span", { class: "pin" }, icon("pin")) : null,
        el("span", { class: "t", title: c.title }, c.title),
        state.busy.has(c.id) ? el("span", { class: "dot", title: "ассистент работает" }) : null,
        el("span", { class: "tools" },
          el("button", { class: "icon-btn", title: c.pinned ? "Открепить" : "Закрепить", onclick: (e) => { e.stopPropagation(); pinChat(c); } }, icon("pin")),
          el("button", { class: "icon-btn", title: "Удалить чат", onclick: (e) => { e.stopPropagation(); deleteChat(c); } }, icon("trash"))));
      box.append(item);
    }
    nav.append(box);
  }
}

async function newProject() {
  const title = prompt("Название проекта (шоу):", "Новое шоу");
  if (!title) return;
  const p = await guard(() => api("POST", "/studio/api/projects", { title }));
  if (p) { await loadSidebar(); newChat(p.id); }
}
async function renameProject(p) {
  const title = prompt("Новое название проекта:", p.title);
  if (title && title !== p.title) { await guard(() => api("PATCH", `/studio/api/projects/${p.id}`, { title })); loadSidebar(); }
}
async function deleteProject(p) {
  if (!confirm(`Удалить проект «${p.title}» и все его чаты?`)) return;
  await guard(() => api("DELETE", `/studio/api/projects/${p.id}`));
  if (state.chat && state.chat.project_id === p.id) state.chatId = null;
  await loadSidebar();
  if (!state.chatId) openFirst();
}
async function pinChat(c) {
  await guard(() => api("PATCH", `/studio/api/chats/${c.id}`, { pinned: !c.pinned }));
  loadSidebar();
}
async function deleteChat(c) {
  if (!confirm(`Удалить чат «${c.title}»?`)) return;
  await guard(() => api("DELETE", `/studio/api/chats/${c.id}`));
  if (c.id === state.chatId) state.chatId = null;
  await loadSidebar();
  if (!state.chatId) openFirst();
}
async function newChat(projectId) {
  projectId = projectId || state.chat?.project_id || state.projects[0]?.id;
  if (!projectId) projectId = (await api("POST", "/studio/api/projects", { title: "Мои шоу" })).id;
  const c = await guard(() => api("POST", "/studio/api/chats", { project_id: projectId }));
  if (c) { await loadSidebar(); openChat(c.id); $("#input").focus(); }
}
async function openFirst() {
  const last = localStorage.getItem("dg-chat");
  const chat = state.chats.find((c) => c.id === last) || state.chats[0];
  if (chat) openChat(chat.id); else newChat();
}

// -- chat --------------------------------------------------------------------------------
const feed = () => $("#feed-inner");
function scrollDown(force = false) {
  const f = $("#feed");
  if (force || f.scrollHeight - f.scrollTop - f.clientHeight < 240) requestAnimationFrame(() => { f.scrollTop = f.scrollHeight; });
}

async function openChat(id) {
  const data = await guard(() => api("GET", `/studio/api/chats/${id}`));
  if (!data) return;
  state.chatId = id;
  state.chat = data.chat;
  state.pending = data.pending;
  state.cards = new Map();
  state.current = null;
  localStorage.setItem("dg-chat", id);
  $("#app").classList.remove("show-sidebar");  // a drawer closes after a chat is chosen
  $("#chat-title").textContent = data.chat.title;
  $("#chat-meta").textContent = data.chat.project_title;
  renderPrecision();
  renderSidebar();
  renderHistory(data);
  Figure.reset();
  Figure.fromAssets(data.assets);
  setBusy(data.busy);
  connectChat(id);
  scrollDown(true);
}

function emptyChat() {
  const tries = ["Беркут с раскрытыми крыльями", "Домбра для финала шоу", "Тулпар в прыжке, вид сбоку", "Шанырак как эмблема"];
  return el("div", { class: "empty-chat" },
    el("div", { class: "brand", style: "justify-content:center" }, el("span", { class: "logo" }, icon("logo"))),
    el("h2", {}, "Какую фигуру сделаем?"),
    el("div", {}, "Опишите идею простыми словами. Ассистент уточнит детали, нарисует варианты, сделает точную 3D-модель и отправит её в Blender."),
    el("div", { class: "suggestions" }, tries.map((t) => el("button", { class: "btn", onclick: () => { $("#input").value = t; send(); } }, t))));
}

function renderHistory(data) {
  const f = feed();
  f.replaceChildren();
  const results = new Map();
  for (const m of data.messages) if (m.role === "user") for (const b of m.content) if (b.type === "tool_result") results.set(b.tool_use_id, b);
  if (!data.messages.length) f.append(emptyChat());
  for (const m of data.messages) {
    if (m.role === "notice") { notice(m.level, m.text); continue; }
    if (m.role === "user") {
      const answer = m.content.find((b) => b.type === "tool_result" && b.ui?.kind === "answer");
      const plain = m.content.filter((b) => b.type !== "tool_result");
      if (answer) f.append(answerBubble(answer));
      if (plain.length) f.append(userBubble(plain, m.answers));
      continue;
    }
    const bubble = new AssistantBubble();
    bubble.fill(m, results);
    f.append(bubble.node);
  }
  if (data.pending) f.append(questionsBlock(data.pending.questions, data.pending.tool_use_id));
  if (data.live) {
    state.current = new AssistantBubble(true);
    if (data.live.notes) state.current.note(0, data.live.notes);
    if (data.live.text) state.current.text(data.live.text);
    f.append(state.current.node);
  }
}

function userBubble(blocks, answers) {
  const pics = blocks.filter((b) => b.type === "image");
  const text = blocks.filter((b) => b.type === "text" && !/^\(картинка пользователя/.test(b.text)).map((b) => b.text).join("\n");
  const flat = (answers || []).flat().filter(Boolean);
  return el("div", { class: "msg user" },
    pics.length ? el("div", { class: "pics" }, pics.map((p) => el("img", { src: chatImageUrl(p.file), onclick: () => openLightbox(chatImageUrl(p.file)) }))) : null,
    flat.length ? el("div", { class: "answers" }, flat.map((a) => el("span", { class: "chip accent" }, a))) : null,
    text ? el("div", { class: "bubble", html: md(text) }) : null);
}
// the user's reply to a question: the chosen options, the typed text and the attached pictures
function answerBubble(answer) {
  return userBubble([...(answer.content || []).filter((b) => b.type === "image"), { type: "text", text: answer.ui.text || "" }], answer.ui.answers);
}

class AssistantBubble {
  constructor(streaming = false) {
    this.node = el("div", { class: "msg assistant" });
    this.notesBox = el("div", { class: "notes" });
    this.textBox = el("div", { class: "bubble" });
    this.toolsBox = el("div", { style: "display:flex;flex-direction:column;gap:10px" });
    this.node.append(this.notesBox, this.textBox, this.toolsBox);
    this.raw = "";
    this.notes = new Map();
    this.typing = streaming ? el("div", { class: "typing" }, el("i"), el("i"), el("i")) : null;
    if (this.typing) this.node.append(this.typing);
  }
  text(delta) {
    this.raw += delta;
    this.textBox.innerHTML = md(this.raw);
    this.stopTyping();
  }
  note(index, delta) {
    let n = this.notes.get(index);
    if (!n) { n = el("div", { class: "note" }); this.notes.set(index, n); this.notesBox.append(n); }
    n.textContent += delta;
  }
  stopTyping() { if (this.typing) { this.typing.remove(); this.typing = null; } }
  tool(use) {
    if (use.name === "ask_user") return null;
    let card = state.cards.get(use.id);
    if (!card) { card = new ToolCard(use.id, use.name); state.cards.set(use.id, card); this.toolsBox.append(card.node); }
    this.stopTyping();
    return card;
  }
  fill(message, results) {
    this.raw = message.content.filter((b) => b.type === "text").map((b) => b.text).join("\n\n");
    this.textBox.innerHTML = md(this.raw);
    this.notesBox.replaceChildren();
    message.content.filter((b) => b.type === "note").forEach((b, i) => this.note(`n${i}`, b.text));
    for (const b of message.content.filter((b) => b.type === "tool_use")) {
      if (b.name === "ask_user") {
        const res = results?.get(b.id);
        if (res) this.toolsBox.append(questionsBlock(b.input.questions || [], b.id, res.ui?.answers || []));
        continue;
      }
      const card = this.tool(b);
      card.input(b.input);
      const res = results?.get(b.id);
      if (res) card.finish(!res.is_error, res.ui, res.content?.map((c) => c.text || "").join("\n"));
      else if (state.pending?.tool_use_id === b.id && state.pending.ui) card.finish(true, state.pending.ui);  // a tool waiting for the user's answer
    }
    this.stopTyping();
  }
}

const TOOL_TITLES = { look_up: ["search", "Справка"], write_brief: ["brief", "Бриф"], generate_images: ["image", "Картинки"], make_3d: ["cube", "3D-модель"], build_exact: ["cube", "Точная модель"], send_to_blender: ["blender", "В Blender"] };

class ToolCard {
  constructor(id, name) {
    this.id = id;
    this.name = name;
    const [ic, title] = TOOL_TITLES[name] || ["gear", name];
    this.stateEl = el("span", { class: "state" }, "…");
    this.head = el("div", { class: "card-head" }, icon(ic), el("span", {}, title), el("span", { class: "spin" }), this.stateEl);
    this.body = el("div", { class: "card-body" });
    this.bar = el("div", { class: "progress" }, el("div"));
    this.stage = el("div", { class: "stage" });
    this.node = el("div", { class: "card" }, this.head, this.body);
    this.items = new Map();
  }
  input(inp) {
    this.inp = inp;
    if (this.name === "look_up") this.stateEl.textContent = inp.name || "";
    if (this.name === "generate_images") this.stateEl.textContent = `${inp.count || 2} вар. · ${({ sheet: "лист видов", render: "объёмный", flat: "плоский" })[inp.style] || ""}`;
    if (this.name === "build_exact") this.stateEl.textContent = ({ dombra: "домбра", shanyrak: "шанырак", yurt: "юрта", baiterek: "Байтерек" })[inp.template] || inp.template;
    if (this.name === "make_3d") this.stateEl.textContent = ({ hunyuan_mv: "Hunyuan3D по карточкам", hunyuan: "Hunyuan3D", triposr: "TripoSR", silhouette: "силуэт" })[inp.method] || "";
  }
  progress(p, stage, label) {
    this.node.classList.add("open");
    if (!this.bar.isConnected) this.body.prepend(this.bar, this.stage);
    $("div", this.bar).style.width = `${Math.round((p || 0) * 100)}%`;
    this.stage.textContent = [label, stage].filter(Boolean).join(": ");
  }
  item(ui) {
    if (ui.kind !== "image") return;
    this.node.classList.add("open");
    this.gallery = this.gallery || el("div", { class: "variants" + (ui.style === "sheet" ? " sheet" : "") });
    if (!this.gallery.isConnected) this.body.append(this.gallery);
    const tile = variantTile(ui), old = this.items.get(ui.file);
    if (old) old.replaceWith(tile); else this.gallery.append(tile);
    this.items.set(ui.file, tile);
  }
  finish(ok, ui, text) {
    this.bar.remove(); this.stage.remove();
    this.node.classList.add(ok ? "done" : "failed", "open");
    if (!ok) { this.stateEl.textContent = "ошибка"; this.body.replaceChildren(el("div", { class: "err-text" }, ui?.text || text || "ошибка")); return; }
    if (!ui) return;
    if (ui.kind === "reference") {
      this.stateEl.textContent = ui.known ? "есть в справке студии" : ui.source === "wikipedia" ? "Википедия" : "ваше описание";
      this.body.replaceChildren(referenceView(ui));
    } else if (ui.kind === "lookup" || (ui.kind === "answer" && ui.lookup)) {  // not found: the user was asked
      const miss = ui.lookup || ui;
      this.stateEl.textContent = "не найдено";
      this.node.classList.add("missed");
      this.body.replaceChildren(el("div", { class: "stage" }, `Не нашёл, как выглядит «${miss.name}»: ${miss.why}.`));
      if (ui.lookup) this.body.append(questionsBlock(ui.lookup.questions || [], this.id, ui.answers || []));
    } else if (ui.kind === "brief") {
      this.stateEl.textContent = ui.title_ru || "";
      this.body.replaceChildren(
        el("div", { class: "brief-title" }, ui.title_ru),
        el("div", { class: "chips" },
          el("span", { class: "chip accent" }, ({ sheet: "лист видов", render: "объёмный рендер", flat: "плоский силуэт" })[ui.style] || ui.style),
          el("span", { class: "chip" }, ({ hunyuan_mv: "3D по карточкам", hunyuan: "3D по картинке", silhouette: "силуэт → объём" })[ui.method] || ui.method),
          ui.symmetry ? el("span", { class: "chip" }, "симметрия") : null,
          el("span", { class: "chip" }, ui.view),
          ui.template ? el("span", { class: "chip ok", title: "для этого предмета есть точный чертёж" }, "есть точный чертёж") : null,
          (ui.references || []).map((t) => el("span", { class: "chip ok", title: "описание написано по справке" }, `по справке: ${t}`))),
        ui.notes_ru ? el("div", {}, ui.notes_ru) : null,
        el("ul", { class: "checks" }, (ui.checks || []).map((c) => el("li", {}, c))),
        el("details", {}, el("summary", {}, "английский промпт"), el("pre", {}, ui.prompt)));
      Figure.brief(ui);
    } else if (ui.kind === "images") {
      this.stateEl.textContent = `${ui.items.length} вар.`;
      for (const it of ui.items) this.item({ kind: "image", ...it });
      const lastSheet = [...ui.items].reverse().find((it) => it.cards);
      if (lastSheet) Figure.cards(lastSheet.cards);
    } else if (ui.kind === "mesh") {
      const q = ui.quality || {};
      this.stateEl.textContent = "готово";
      this.body.replaceChildren(
        el("img", { class: "mesh-preview", src: fileUrl(ui.files.preview), onclick: () => openLightbox(fileUrl(ui.files.preview)) }),
        el("div", { class: "chips" },
          q.depth_ratio !== undefined ? el("span", { class: "chip" }, `толщина ${q.depth_ratio.toFixed(2)}`) : null,
          q.front_iou !== undefined && q.front_iou !== null ? el("span", { class: "chip" }, `силуэт ${q.front_iou.toFixed(2)}`) : null,
          q.faces ? el("span", { class: "chip" }, `${(q.faces / 1000).toFixed(0)} тыс. граней`) : null,
          (ui.warnings || []).length ? el("span", { class: "chip warn", title: ui.warnings.join("\n") }, "есть предупреждения") : null),
        el("div", { class: "actions" },
          el("button", { class: "btn small", onclick: () => Figure.mesh(ui) }, icon("cube"), "Показать в 3D"),
          el("button", { class: "btn small", onclick: () => toBlender(ui.files.mesh) }, icon("blender"), "В Blender"),
          el("a", { class: "btn small", href: fileUrl(ui.files.mesh), download: ui.files.mesh }, icon("down"), "GLB")));
      Figure.mesh(ui);
    } else if (ui.kind === "blender") {
      this.stateEl.textContent = "в очереди";
      this.inbox = ui.inbox;
      this.body.replaceChildren(el("div", {}, `«${ui.name}» ждёт в очереди: Blender с аддоном DroneGen заберёт её сам.`));
    }
  }
}

// how a named character looks: the picture and the intro of its article (or the user's own description)
function referenceView(ui) {
  if (ui.known) return el("div", {}, `«${ui.name}» — в справке студии, искать в интернете не нужно.`);
  return el("div", { class: "reference" },
    ui.image ? el("img", { src: fileUrl(ui.image), title: ui.title, onclick: () => openLightbox(fileUrl(ui.image)) }) : null,
    el("div", { class: "ref-body" },
      el("div", { class: "brief-title" }, ui.title),
      ui.text ? el("div", { class: "ref-text", title: ui.text }, ui.text) : null,
      el("div", { class: "chips" },
        el("span", { class: "chip ok", title: "описание для картинок напишется по этой справке" }, "попадёт в бриф"),
        ui.url ? el("a", { class: "chip", href: ui.url, target: "_blank", rel: "noopener" }, "статья в Википедии") : null)));
}

function checkBadge(check) {
  if (check === undefined) return null;
  if (!check) return el("span", { class: "chip", title: "арт-директор не смог проверить картинку" }, "не проверено");
  const notes = check.checks.map((c) => `${c.passed ? "✓" : "✗"} ${c.check}: ${c.note}`).join("\n");
  const failed = check.checks.filter((c) => !c.passed);
  return el("span", { class: "chip " + (check.ok ? "ok" : "warn"), title: `Видно: ${check.seen}\n${notes}` },
    check.ok ? "✓ проверка пройдена" : `✗ ${failed.map((c) => c.note).join("; ")}`);
}

function variantTile(it) {
  const cards = it.cards ? el("div", { class: "mini-cards" }, Object.values(it.cards).map((c) => el("img", { src: fileUrl(c), title: "карточка вида", onclick: () => openLightbox(fileUrl(c)) }))) : null;
  return el("div", { class: "variant" },
    el("img", { src: fileUrl(it.file), loading: "lazy", onclick: () => openLightbox(fileUrl(it.file)) }),
    el("div", { class: "row" },
      el("span", { class: "n", title: it.seed !== undefined ? `seed ${it.seed}` : "" }, `Вариант ${it.n}`),
      el("span", { class: "s" }),
      el("button", { class: "btn small", title: "Сделать 3D из этого варианта", onclick: () => quick(`Беру вариант ${it.n}, делай 3D`) }, "В 3D")),
    it.cards_error ? el("div", { class: "stage", style: "padding:0 8px 8px" }, it.cards_error) : cards,
    it.cards_note ? el("div", { class: "stage", style: "padding:0 8px 8px" }, it.cards_note) : null,
    it.check !== undefined ? el("div", { style: "padding:0 8px 8px" }, checkBadge(it.check)) : null);
}

function questionsBlock(questions, toolUseId, answered) {
  const chosen = questions.map(() => new Set());
  const done = !!answered;
  const box = el("div", { class: "questions" + (done ? " answered" : "") });
  questions.forEach((q, i) => {
    const opts = el("div", { class: "options" });
    for (const o of q.options) {
      const on = done && (answered[i] || []).includes(o);
      const b = el("button", { class: "option" + (on ? " on" : ""), disabled: done || null }, o);
      b.addEventListener("click", () => {
        // «Опишу словами» / «Прикреплю картинку»: the answer is what the user types or attaches, not the button
        const action = q.actions?.[o];
        if (action === "describe") { input.placeholder = "Опишите, как это выглядит: форма, цвета, детали…"; input.focus(); return; }
        if (action === "attach") { $("#file").click(); input.placeholder = "Можно добавить пару слов к картинке и отправить"; return; }
        if (!q.multiple) { chosen[i].clear(); opts.querySelectorAll(".option").forEach((x) => x.classList.remove("on")); }
        if (chosen[i].has(o)) { chosen[i].delete(o); b.classList.remove("on"); } else { chosen[i].add(o); b.classList.add("on"); }
        if (questions.length === 1 && !q.multiple) submit();
      });
      opts.append(b);
    }
    box.append(el("div", {}, el("div", { class: "q" }, q.question), opts));
  });
  const submit = () => {
    const answers = chosen.map((s) => [...s]);
    send({ answers });
  };
  if (!done) box.append(el("div", { class: "foot" }, "Выберите вариант или напишите свой ответ ниже",
    questions.length > 1 || questions.some((q) => q.multiple) ? el("button", { class: "btn primary small", onclick: submit }, "Ответить") : null));
  box.dataset.toolUseId = toolUseId;
  return box;
}

function notice(level, text) {
  // the first line is the message, the rest (a log excerpt) goes under a fold
  const [head, ...rest] = String(text).split("\n");
  const box = el("div", { class: `notice ${level}` + (rest.length ? " long" : "") }, head);
  if (rest.length) box.append(el("details", {}, el("summary", {}, "подробности"), el("pre", {}, rest.join("\n"))));
  feed().append(box);
  scrollDown();
}

// -- live events -------------------------------------------------------------------------
function connectChat(id) {
  if (state.es) state.es.close();
  const es = new EventSource(withToken(`/studio/api/chats/${id}/events`));
  state.es = es;
  es.onmessage = (m) => { if (state.chatId === id) onEvent(JSON.parse(m.data)); };
  es.addEventListener("hello", (m) => { const h = JSON.parse(m.data); setBusy(h.busy); });
}

function bubble() {
  if (!state.current) { state.current = new AssistantBubble(true); feed().append(state.current.node); }
  return state.current;
}

function onEvent(e) {
  const f = feed();
  $(".empty-chat", f)?.remove();
  switch (e.type) {
    case "user_message": {
      const m = e.message;
      const answer = m.content.find((b) => b.type === "tool_result" && b.ui?.kind === "answer");
      document.querySelectorAll(".questions:not(.answered)").forEach((q) => q.classList.add("answered"));
      input.placeholder = INPUT_PLACEHOLDER;
      if (answer) f.append(answerBubble(answer));
      const plain = m.content.filter((b) => b.type !== "tool_result");
      if (plain.length) f.append(userBubble(plain, m.answers));
      setBusy(true);
      break;
    }
    case "assistant_start":
      if (state.current && !state.current.raw && !state.current.toolsBox.children.length) state.current.node.remove();
      state.current = new AssistantBubble(true);
      f.append(state.current.node);
      break;
    case "text": bubble().text(e.delta); break;
    case "note": bubble().note(e.index, e.delta); break;
    case "tool_call": bubble().tool({ id: e.id, name: e.name }); break;
    case "tool_running": { const c = bubble().tool({ id: e.id, name: e.name }); c?.input(e.input); c?.progress(0.02, "запуск"); break; }
    case "tool_progress": state.cards.get(e.tool_use_id)?.progress(e.progress, e.stage, e.label); break;
    case "tool_item": state.cards.get(e.tool_use_id)?.item(e.ui); break;
    case "tool_result": state.cards.get(e.tool_use_id)?.finish(e.ok, e.ui, e.text); break;
    case "assistant_message": {
      const b = bubble();
      b.raw = "";
      b.fill(e.message, null);
      state.current = null;
      break;
    }
    case "choices": f.append(questionsBlock(e.questions, e.tool_use_id)); break;
    case "notice": notice(e.level, e.text); break;
    case "provider": setProvider(e.name, e.cloud, state.status?.providers?.mode === "claude" && !e.cloud); break;
    case "chat": if (e.chat.id === state.chatId) $("#chat-title").textContent = e.chat.title; loadSidebar(); break;
    case "inbox": {
      const card = [...state.cards.values()].find((c) => c.inbox === e.item.id);
      if (card) card.stateEl.textContent = ({ queued: "в очереди", taken: "Blender забирает…", imported: "в сцене", failed: "ошибка" })[e.item.status] || e.item.status;
      break;
    }
    case "turn_end":
      if (state.current) { state.current.stopTyping(); if (!state.current.raw && !state.current.toolsBox.children.length && !state.current.notes.size) state.current.node.remove(); }
      state.current = null;
      setBusy(false);
      break;
  }
  scrollDown(e.type === "user_message");  // the user's own message always brings the view down
}

function connectGlobal() {
  const es = new EventSource(withToken("/studio/api/events"));
  state.ges = es;
  es.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e.type === "user_message") { state.busy.add(e.chat_id); renderSidebar(); }
    if (e.type === "turn_end") { state.busy.delete(e.chat_id); renderSidebar(); if (e.chat_id !== state.chatId) loadSidebar(); }
    if (e.type === "notice" && e.chat_id !== state.chatId && /Blender/.test(e.text)) toast(e.text);
  };
}

function setBusy(busy) {
  if (busy) state.busy.add(state.chatId); else state.busy.delete(state.chatId);
  $("#stop").hidden = !busy;
  $("#send").disabled = busy;
  renderSidebar();
}

// -- composer ----------------------------------------------------------------------------
const input = $("#input");
const INPUT_PLACEHOLDER = input.placeholder;
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 220) + "px"; }
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
$("#send").addEventListener("click", () => send());
$("#stop").addEventListener("click", () => guard(() => api("POST", `/studio/api/chats/${state.chatId}/stop`)));
$("#file").addEventListener("change", (e) => { addFiles([...e.target.files]); e.target.value = ""; });
document.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});
$("#feed").addEventListener("dragover", (e) => e.preventDefault());
$("#feed").addEventListener("drop", (e) => { e.preventDefault(); addFiles([...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"))); });

const VIEW_NAMES = { front: "спереди", left: "слева", back: "сзади", right: "справа" };
function addFiles(files) {
  for (const f of files.slice(0, 4 - state.attachments.length)) state.attachments.push({ file: f, view: "" });
  // several photos of one object: suggest the sides in turn, the user corrects them
  if (state.attachments.length > 1) {
    const used = new Set(state.attachments.map((a) => a.view).filter(Boolean));
    for (const a of state.attachments) if (!a.view) { a.view = Object.keys(VIEW_NAMES).find((v) => !used.has(v)) || ""; used.add(a.view); }
  }
  renderAttachments();
}
function renderAttachments() {
  const box = $("#attachments");
  box.replaceChildren(...state.attachments.map((a, i) => {
    const url = URL.createObjectURL(a.file);
    const sel = el("select", { title: "Какую сторону предмета показывает фото", class: a.view ? "" : "unset", onchange: (e) => { a.view = e.target.value; e.target.className = a.view ? "" : "unset"; } },
      el("option", { value: "" }, "ракурс?"), ...Object.entries(VIEW_NAMES).map(([v, n]) => el("option", { value: v, selected: v === a.view }, n)));
    return el("div", { class: "att" }, el("img", { src: url }), sel,
      el("button", { title: "Убрать", onclick: () => { state.attachments.splice(i, 1); renderAttachments(); } }, "×"));
  }));
}

function renderPrecision() {
  const p = state.chat?.state?.precision || "auto";
  for (const b of document.querySelectorAll("#precision [data-p]")) b.classList.toggle("on", b.dataset.p === p);
}
$("#precision").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-p]");
  if (!b || !state.chatId) return;
  const chat = await guard(() => api("PATCH", `/studio/api/chats/${state.chatId}`, { precision: b.dataset.p }));
  if (chat) { state.chat = { ...state.chat, state: chat.state }; renderPrecision(); }
});

async function send(extra = {}) {
  if (!state.chatId) return;
  const text = extra.text ?? input.value;
  if (!text.trim() && !state.attachments.length && !extra.answers) return;
  if (state.busy.has(state.chatId)) { toast("Ассистент ещё отвечает — дождитесь или нажмите «Стоп»"); return; }
  const form = new FormData();
  form.append("text", text);
  if (extra.answers) form.append("answers", JSON.stringify(extra.answers));
  for (const a of state.attachments) form.append("files", a.file, a.file.name);
  if (state.attachments.some((a) => a.view)) form.append("views", JSON.stringify(state.attachments.map((a) => a.view)));
  const ok = await guard(() => api("POST", `/studio/api/chats/${state.chatId}/messages`, form, true));
  if (ok) {
    if (extra.text === undefined) { input.value = ""; autosize(); }
    state.attachments = [];
    renderAttachments();
  }
}
function quick(text) { send({ text }); }

async function toBlender(mesh) {
  if (!mesh) return toast("Сначала сделайте 3D-модель");
  const title = state.chat?.title || "Фигура";
  const item = await guard(() => api("POST", `/studio/api/chats/${state.chatId}/blender`, { file: mesh, name: title }));
  if (item) toast(`«${item.name}» в очереди для Blender`);
}

// -- the figure panel --------------------------------------------------------------------
const Figure = {
  reset() {
    state.lastMesh = null;
    const mv = $("#mv");
    mv.hidden = true; mv.removeAttribute("src");
    $("#viewer-empty").hidden = false; $("#viewer-hint").hidden = true;
    $("#to-blender").disabled = true;
    for (const id of ["#dl-glb", "#dl-obj"]) { $(id).removeAttribute("href"); $(id).setAttribute("aria-disabled", "true"); }
    for (const id of ["#fig-metrics", "#fig-cards", "#fig-brief", "#fig-params"]) $(id).hidden = true;
    $("#fig-title").textContent = "";
    state.template = null;
  },
  fromAssets(assets) {
    const last = (kind) => [...assets].reverse().find((a) => a.kind === kind);
    const brief = last("brief");
    if (brief) this.brief(brief.data);
    const sheet = [...assets].reverse().find((a) => a.kind === "image" && a.data.cards);
    if (sheet) this.cards(sheet.data.cards);
    const mesh = last("mesh");
    if (mesh) this.mesh({ files: mesh.files, quality: mesh.data.quality, template: mesh.data.template, values: mesh.data.values });
    else if (brief?.data.template) this.params(brief.data.template, null);
  },
  mesh(ui) {
    state.lastMesh = ui.files.mesh;
    const mv = $("#mv");
    mv.src = fileUrl(ui.files.mesh);
    mv.hidden = false;
    $("#viewer-empty").hidden = true; $("#viewer-hint").hidden = false;
    $("#to-blender").disabled = false;
    const glb = $("#dl-glb"), obj = $("#dl-obj");
    glb.href = fileUrl(ui.files.mesh); glb.download = ui.files.mesh; glb.removeAttribute("aria-disabled");
    if (ui.files.mesh_obj) { obj.href = fileUrl(ui.files.mesh_obj); obj.download = ui.files.mesh_obj; obj.removeAttribute("aria-disabled"); }
    const q = ui.quality || {};
    const box = $("#fig-metrics .metrics");
    box.replaceChildren(
      metric(q.depth_ratio?.toFixed(2), "толщина / ширина"),
      metric(q.front_iou !== undefined && q.front_iou !== null ? q.front_iou.toFixed(2) : "—", "совпадение спереди"),
      metric(q.faces ? `${(q.faces / 1000).toFixed(0)}k` : "—", "треугольников"),
      metric(q.asymmetry_pct !== undefined ? `${q.asymmetry_pct.toFixed(1)}%` : "—", "асимметрия"));
    $("#fig-metrics").hidden = false;
    if (ui.template) this.params(ui.template, ui.values);
    else $("#fig-params").hidden = true;
  },
  // sliders of an exact drawing; values === null: nothing built yet
  async params(key, values) {
    if (!state.templates) state.templates = await guard(() => api("GET", "/studio/api/templates"));
    const t = state.templates?.[key];
    if (!t) return;
    state.template = { key, values: { ...(values || {}) } };
    $("#params-title").textContent = `· ${t.title}`;
    $("#params-about").textContent = t.about;
    const list = $("#params-list");
    list.replaceChildren(...Object.entries(t.params).map(([name, p]) => {
      const cur = values?.[name] ?? p.default;
      const step = p.integer ? 1 : Math.max((p.max - p.min) / 200, 0.001);
      const out = el("output", {}, fmt(cur, p));
      const input = el("input", { type: "range", min: p.min, max: p.max, step, value: cur });
      input.addEventListener("input", () => { out.textContent = fmt(+input.value, p); });
      input.addEventListener("change", () => rebuild(name, +input.value, input.closest(".param")));
      return el("div", { class: "param" }, el("label", {}, p.title), out, input);
    }));
    $("#params-build").textContent = values ? "Перестроить" : "Построить по чертежу";
    $("#fig-params").hidden = false;
  },
  cards(cards) {
    const names = { front: "спереди", side: "сбоку", back: "сзади", left: "слева", right: "справа" };
    $("#fig-cards .views").replaceChildren(...Object.entries(cards).map(([v, f]) =>
      el("figure", {}, el("img", { src: fileUrl(f), onclick: () => openLightbox(fileUrl(f)) }), el("figcaption", {}, names[v] || v))));
    $("#fig-cards").hidden = false;
  },
  brief(b) {
    $("#fig-title").textContent = b.title_ru ? `· ${b.title_ru}` : "";
    $("#brief-chips").replaceChildren(
      el("span", { class: "chip accent" }, ({ sheet: "лист видов", render: "объёмный", flat: "плоский" })[b.style] || b.style),
      el("span", { class: "chip" }, b.view || ""),
      b.template ? el("span", { class: "chip ok", title: "для этого предмета есть точный чертёж" }, "есть точный чертёж") : null);
    $("#brief-subject").value = b.subject || "";
    $("#brief-checks").replaceChildren(...(b.checks || []).map((c) => el("li", {}, c)));
    $("#fig-brief").hidden = false;
  },
};
const fmt = (v, p) => `${p.integer ? Math.round(v) : +v.toFixed(3)}${p.unit ? " " + p.unit : ""}`;
let rebuildTimer = null;
async function rebuild(name, value, row) {
  if (!state.template || !state.chatId) return;
  if (name) state.template.values[name] = value;
  clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(async () => {
    row?.classList.add("busy");
    const r = await guard(() => api("POST", `/studio/api/chats/${state.chatId}/build`, { template: state.template.key, values: state.template.values }));
    row?.classList.remove("busy");
    if (r) Figure.mesh({ files: r.files, quality: r.quality, template: r.template, values: r.values });
  }, 150);
}
$("#params-build").addEventListener("click", () => rebuild(null, 0, null));
$("#params-reset").addEventListener("click", () => { if (state.template) { state.template.values = {}; rebuild(null, 0, null); } });
function metric(value, label) { return el("div", { class: "metric" }, el("b", {}, value ?? "—"), el("span", {}, label)); }
$("#to-blender").addEventListener("click", () => toBlender(state.lastMesh));
$("#viewer-reset").addEventListener("click", () => { const mv = $("#mv"); mv.cameraOrbit = "0deg 75deg 105%"; mv.fieldOfView = "auto"; mv.jumpCameraToGoal?.(); });
$("#redraw").addEventListener("click", () => {
  const s = $("#brief-subject").value.trim();
  if (s) quick(`Перерисуй с таким английским описанием (subject), остальное из брифа:\n${s}`);
});

// -- title, settings, shortcuts ----------------------------------------------------------
const title = $("#chat-title");
title.addEventListener("click", () => { title.contentEditable = "true"; title.focus(); });
title.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); title.blur(); } });
title.addEventListener("blur", async () => {
  title.contentEditable = "false";
  const t = title.textContent.trim();
  if (state.chat && t && t !== state.chat.title) { await guard(() => api("PATCH", `/studio/api/chats/${state.chatId}`, { title: t })); state.chat.title = t; loadSidebar(); }
});

// cloud: Claude answers; fallback: the cloud was chosen but cannot be used, the local model answers
function setProvider(name, cloud, fallback) {
  const badge = $("#provider-badge");
  badge.className = cloud ? "cloud" : "local";
  $(".label", badge).textContent = cloud ? "облако: Claude" : "локально";
  if (fallback !== undefined) $("#offline-chip").classList.toggle("on", !!fallback);
}

async function loadStatus() {
  state.status = await api("GET", "/studio/api/status");
  const p = state.status.providers;
  setProvider(p.current, p.cloud, p.fallback);
  return state.status;
}

const dlg = $("#settings");
let chosenProvider = "auto";
$("#open-settings").addEventListener("click", async () => {
  const st = await guard(loadStatus);
  if (!st) return;
  chosenProvider = st.providers.mode;
  for (const b of dlg.querySelectorAll("#set-provider button")) b.classList.toggle("on", b.dataset.v === chosenProvider);
  showCloudFields();
  const p = st.providers;
  const line = (id, ok, text) => { const n = $(id); n.className = "status-line " + (ok ? "ok" : "no"); $("span:last-child", n).textContent = text; };
  line("#st-claude", p.claude.available, p.claude.available ? `Claude готов (${p.claude.model})` : `Claude: ${p.claude.reason}`);
  line("#st-local", p.local.available, p.local.available ? `Локальная модель установлена${p.local.running ? ", запущена" : ""}` : `Локальная модель: ${p.local.reason}`);
  $("#set-model").replaceChildren(...st.claude_models.map((m) => el("option", { value: m, selected: m === st.settings.claude_model }, m)));
  $("#set-images").value = st.settings.image_backend;
  $("#set-web").checked = st.settings.web_lookup !== false;
  $("#key-state").textContent = st.secrets.anthropic_api_key.set ? `(задан ${st.secrets.anthropic_api_key.hint})` : "(не задан)";
  $("#sk-state").textContent = st.secrets.sketchfab_token.set ? `(задан ${st.secrets.sketchfab_token.hint})` : "";
  $("#set-key").value = ""; $("#set-token").value = "";
  dlg.showModal();
});
function showCloudFields() { dlg.querySelectorAll(".cloud-only").forEach((n) => { n.hidden = chosenProvider !== "claude"; }); }
dlg.querySelectorAll("#set-provider button").forEach((b) => b.addEventListener("click", () => {
  chosenProvider = b.dataset.v;
  dlg.querySelectorAll("#set-provider button").forEach((x) => x.classList.toggle("on", x === b));
  showCloudFields();
}));
dlg.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => dlg.close()));
$("#key-clear").addEventListener("click", async () => {
  if (!confirm("Удалить сохранённый ключ Claude API?")) return;
  await guard(() => api("PUT", "/studio/api/settings", { anthropic_api_key: "" }));
  $("#key-state").textContent = "(не задан)";
});
$("#settings-save").addEventListener("click", async () => {
  const body = { provider: chosenProvider, claude_model: $("#set-model").value, image_backend: $("#set-images").value,
                 web_lookup: $("#set-web").checked };
  if ($("#set-key").value.trim()) body.anthropic_api_key = $("#set-key").value.trim();
  if ($("#set-token").value.trim()) body.sketchfab_token = $("#set-token").value.trim();
  const st = await guard(() => api("PUT", "/studio/api/settings", body));
  if (st) { setProvider(st.providers.current, st.providers.cloud, st.providers.fallback); dlg.close(); toast("Настройки сохранены"); }
});

$("#server-quit").addEventListener("click", async () => {
  if (!confirm("Выключить сервер DroneGen? Идущие задачи прервутся, видеопамять освободится.")) return;
  if (await guard(() => api("POST", "/shutdown"))) {
    dlg.close();
    document.body.innerHTML = '<p style="padding:40px;font:16px system-ui;color:#999">Сервер выключен. Окно можно закрыть. Запустить снова: ярлык «DroneGen Studio» или Alt+Shift+D в Blender.</p>';
    setTimeout(() => window.close(), 800);
  }
});

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "k") { e.preventDefault(); newChat(); }
  if (e.ctrlKey && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "b") { e.preventDefault(); toBlender(state.lastMesh); }
  if (e.key === "Escape") $("#lightbox").classList.remove("on");
});
const app = $("#app");
const drawer = (name) => app.classList.toggle(`show-${name}`);
$("#sidebar-btn").addEventListener("click", () => { app.classList.remove("show-figure"); drawer("sidebar"); });
$("#figure-btn").addEventListener("click", () => { app.classList.remove("show-sidebar"); drawer("figure"); });
$("#scrim").addEventListener("click", () => app.classList.remove("show-sidebar", "show-figure"));
$("#new-chat").addEventListener("click", () => newChat());
$("#new-project").addEventListener("click", newProject);
let searchTimer;
$("#search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadSidebar, 250); });

// -- start -------------------------------------------------------------------------------
(async function start() {
  try {
    await loadStatus();
    await loadSidebar();
    if (!state.projects.length) await api("POST", "/studio/api/projects", { title: "Мои шоу" }).then(loadSidebar);
    connectGlobal();
    await openFirst();
  } catch (e) {
    feed().replaceChildren(el("div", { class: "empty-chat" }, el("h2", {}, "Нет связи с сервером DroneGen"), el("div", {}, e.message)));
  }
})();
