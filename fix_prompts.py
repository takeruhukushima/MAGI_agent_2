import os
import re
from pathlib import Path

def fix_prompt_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match prompt template usage
    pattern = r'prompt_template\s*=\s*PromptManager\.get_prompt\([^)]+\)\s*\n\s*self\.prompt\s*=\s*ChatPromptTemplate\.from_template\(prompt_template\.template\)'
    
    # Replacement pattern
    replacement = (
        '# Get the prompt template string from PromptManager\n'
        '        prompt_template = PromptManager.get_prompt\\1\n'
        '        self.prompt = ChatPromptTemplate.from_template(prompt_template)'
    )
    
    # Replace the pattern
    new_content = re.sub(pattern, replacement, content)
    
    # Write back if changes were made
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    # Find all Python files in the chains directory
    chains_dir = Path('my_agent/chains')
    python_files = list(chains_dir.glob('**/*.py'))
    
    fixed_files = 0
    
    for file_path in python_files:
        try:
            if fix_prompt_template(file_path):
                print(f"Fixed: {file_path}")
                fixed_files += 1
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"\nFixed {fixed_files} files.")

if __name__ == "__main__":
    main()
