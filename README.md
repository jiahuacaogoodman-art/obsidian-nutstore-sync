# 果札：AI pro

果札：AI pro 是一个面向 Obsidian 的社区增强插件，把 WebDAV 同步、AI 对话、多模态输入、图像生成和 Vault 文件操作放在同一个工作流里。

_Guozha: AI pro is a community-enhanced Obsidian plugin that combines WebDAV sync, AI chat, multimodal input, image generation, and vault file operations in one workflow._

> 本项目基于 [nutstore/obsidian-nutstore-sync](https://github.com/nutstore/obsidian-nutstore-sync) 开发，遵循 AGPL-3.0 协议。果札：AI pro 不是坚果云官方插件。
>
> This project is derived from [nutstore/obsidian-nutstore-sync](https://github.com/nutstore/obsidian-nutstore-sync) and remains licensed under AGPL-3.0. Guozha: AI pro is not an official Nutstore plugin.

---

## 主要特性 | Key Features

- **AI 对话 | AI Chat**
  在 Obsidian 内直接打开对话面板，使用兼容 OpenAI 接口的模型。
  _Chat inside Obsidian with OpenAI-compatible providers._

- **流式输出 | Streaming Output**
  桌面端使用 Node 网络通道绕过浏览器 CORS 限制，让回复实时显示在对话框中。
  _Desktop streaming uses Node transport to avoid browser CORS issues and show replies live._

- **多模态输入 | Multimodal Input**
  支持在聊天中附加图片，让模型基于文本和图像一起回答。
  _Attach images in chat so models can respond with text and visual context._

- **图像生成 | Image Generation**
  支持图像生成模型，并将生成结果保存到 Vault 后直接显示在对话框里。
  _Generate images, save them into the vault, and preview them inline in chat._

- **输出参数调节 | Generation Controls**
  支持调节 temperature 和最大输出长度。
  _Tune temperature and max output length._

- **Vault 文件助手 | Vault Agent**
  AI 可在授权后读取、编辑和管理 Vault 中的文件，并支持工具调用。
  _The assistant can read, edit, and manage vault files after permission checks._

- **WebDAV 同步 | WebDAV Sync**
  保留原有的双向同步、增量同步、冲突处理、过滤器、大文件跳过、日志和缓存管理能力。
  _Keeps the original WebDAV sync workflow: two-way sync, incremental updates, conflict handling, filters, large-file skipping, logs, and cache management._

---

## 使用方式 | Setup

1. 在 Obsidian 插件目录中安装 `main.js`、`styles.css`、`manifest.json`。
2. 启用 **果札：AI pro**。
3. 在插件设置中配置 WebDAV 同步信息和 AI Provider。
4. 从命令面板或侧边栏打开 **果札：AI pro** 对话框。

_Install `main.js`, `styles.css`, and `manifest.json` into an Obsidian plugin folder, enable **Guozha: AI pro**, configure WebDAV and AI providers, then open the chat panel from the command palette or sidebar._

---

## 注意事项 | Notes

- 同步前请先备份 Vault。
- AI 工具操作可能修改文件，建议先保持权限确认开启。
- 若使用兼容 OpenAI 的第三方接口，请确认其支持对应模型能力。
- 本项目为社区改版，不代表坚果云官方立场。

_Back up your vault before syncing. AI tool operations may modify files, so permission confirmation is recommended. For third-party OpenAI-compatible endpoints, verify the model capabilities you plan to use. This is a community fork, not an official Nutstore release._

---

## License

AGPL-3.0. Original upstream work by [nutstore/obsidian-nutstore-sync](https://github.com/nutstore/obsidian-nutstore-sync).