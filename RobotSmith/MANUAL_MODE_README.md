# Manual RobotSmith Mode

This simplified pipeline removes API calls and the critic loop, allowing you to manually interact with Gemini.

## Usage

```bash
python3 manual_pipeline.py --task_name task01_calabash --task_prompt_json_dir task01_calabash/task_prompt.json
```

## Workflow

1. **Script outputs a text-only prompt** - No API calls, no 3D scene images
2. **Copy the prompt to Gemini website** manually
3. **Paste Gemini's response** back into the terminal
4. **Press Ctrl+D** (Linux/Mac) or **Ctrl+Z** (Windows) to finish input
5. **Tool is generated** - Creates .obj mesh file

## Output

The script tells you:
- Tool name
- Output directory location
- Generated .obj files
- How to render the tool using trimesh

## Example Render Command

```python
import trimesh
mesh = trimesh.load('path/to/tool.obj')
mesh.show()
```

## Changes from Original

- ❌ No OpenAI/Azure API calls
- ❌ No critic loop (single generation)
- ❌ No 3D scene rendering in prompts
- ❌ No image inputs to LLM
- ✅ Text-only prompts
- ✅ Manual Gemini interaction
- ✅ Single tool generation
- ✅ Clear output instructions
