# PyUT Agent 图标说明

## 当前图标

**文件**: `resources/icon.svg`

**尺寸**: 128x128 (SVG 矢量格式)

**设计说明**:
- 蓝色圆形背景 (#0e639c)
- 白色试管图标（代表测试）
- 绿色液体（代表 AI/生长）
- 金色闪光点（代表智能）
- "PyUT" 文字标识

## 使用方式

### VS Code 插件支持 SVG

VS Code 插件支持 SVG 格式的图标，可以直接使用：

```json
{
  "icon": "resources/icon.svg"
}
```

### 如需 PNG 格式

如果需要 PNG 格式，可以使用以下工具转换：

1. **在线转换**: 
   - https://cloudconvert.com/svg-to-png
   - https://svgtopng.com/

2. **命令行** (需要安装 ImageMagick):
   ```bash
   convert -resize 128x128 resources/icon.svg resources/icon.png
   ```

3. **使用 Node.js**:
   ```bash
   npm install -g svgo pngjs
   svg2png resources/icon.svg -o resources/icon.png -w 128 -h 128
   ```

## 图标尺寸要求

VS Code Marketplace 要求以下尺寸：

- **小图标**: 20x20 (侧边栏)
- **中图标**: 40x40 (详情页面)
- **大图标**: 128x128 (Marketplace 列表)
- **超大图标**: 512x512 (推荐，用于高清显示)

## 建议

**推荐做法**:
1. 保留 SVG 源文件（可无限缩放）
2. 生成 128x128 PNG 用于插件
3. 生成 512x512 PNG 用于 Marketplace

**当前状态**:
- ✅ SVG 源文件已创建
- ⚠️ 需要生成 PNG 版本（可选）

## 替代方案

如果不想生成 PNG，可以：

1. **使用现有图标服务**:
   - 使用 VS Code 内置图标（如 `$(beaker)`）
   - 使用 emoji（不推荐）

2. **临时方案**:
   - 使用 SVG 作为主图标
   - 在 package.json 中指定 SVG 路径

---

**创建时间**: 2026-04-07  
**设计师**: PyUT Team
