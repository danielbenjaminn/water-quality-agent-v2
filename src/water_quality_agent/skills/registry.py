from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    path: Path

class SkillRegistry:
    def __init__(self, root: Path | None = None):
        self.root=root or Path(__file__).resolve().parent
        self.skills=self._load()
    def _load(self):
        out={}
        for p in self.root.glob("*/SKILL.md"):
            text=p.read_text(encoding="utf-8")
            name=re.search(r"^name:\s*(.+)$",text,re.M)
            desc=re.search(r"^description:\s*(.+)$",text,re.M)
            key=(name.group(1).strip() if name else p.parent.name)
            out[key]=Skill(key,desc.group(1).strip() if desc else "",text,p)
        return out
    def catalog(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self.skills.values())
    def get(self,name:str) -> Skill: return self.skills[name]
