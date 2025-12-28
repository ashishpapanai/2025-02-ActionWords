#!/usr/bin/env python3
"""
Example script showing how to use variable_value_mapping.yaml programmatically
"""

import yaml
from pathlib import Path

# Load the mapping
yaml_path = Path(__file__).parent / "variable_value_mapping.yaml"
with open(yaml_path, 'r') as f:
    mapping = yaml.safe_load(f)

def get_code_label(variable_name: str, code: int) -> str:
    """Get the label for a specific code in a variable."""
    var_info = mapping['variables'].get(variable_name)
    if not var_info:
        return None
    
    # Check if it uses a shared scale
    if 'uses_scale' in var_info:
        scale_name = var_info['uses_scale']
        return mapping['shared_scales'][scale_name]['codes'].get(code)
    
    # Otherwise use direct codes
    if 'codes' in var_info and isinstance(var_info['codes'], dict):
        return var_info['codes'].get(code)
    
    return None

def get_missing_label(variable_name: str, missing_code: int) -> str:
    """Get the label for a missing value code."""
    var_info = mapping['variables'].get(variable_name)
    if not var_info:
        return None
    
    # Check if it uses a shared scale
    if 'uses_scale' in var_info:
        scale_name = var_info['uses_scale']
        return mapping['shared_scales'][scale_name]['missing'].get(missing_code)
    
    # Otherwise use direct missing codes
    if 'missing' in var_info and isinstance(var_info['missing'], dict):
        return var_info['missing'].get(missing_code)
    
    return None

def get_variables_by_category(category: str) -> dict:
    """Get all variables in a specific category."""
    return {k: v for k, v in mapping['variables'].items() 
            if v.get('category') == category}

def decode_value(variable_name: str, value) -> str:
    """Decode a raw value to its label."""
    if value is None or (isinstance(value, float) and (value < 0 or value != value)):
        return "Missing"
    
    # Try as integer code
    if isinstance(value, (int, float)) and value == int(value):
        code = int(value)
        
        # Check missing codes first
        missing_label = get_missing_label(variable_name, code)
        if missing_label:
            return f"Missing: {missing_label}"
        
        # Then regular codes
        label = get_code_label(variable_name, code)
        if label:
            return label
    
    # For continuous variables or unrecognized codes
    var_info = mapping['variables'].get(variable_name, {})
    if var_info.get('type') == 'continuous':
        return f"{value} {var_info.get('unit', '')}"
    
    return str(value)

# Example usage
if __name__ == "__main__":
    print("=== Example 1: Decode a specific code ===")
    print(f"task_laundry code 1 = {get_code_label('task_laundry', 1)}")
    print(f"task_laundry code 3 = {get_code_label('task_laundry', 3)}")
    print(f"task_laundry code -9 = {get_missing_label('task_laundry', -9)}")
    
    print("\n=== Example 2: Get all task division variables ===")
    task_vars = get_variables_by_category('behavior_task_division')
    for var_name in sorted(task_vars.keys()):
        print(f"  - {var_name}")
    
    print("\n=== Example 3: Decode values ===")
    print(f"decode_value('task_grocery', 1) = {decode_value('task_grocery', 1)}")
    print(f"decode_value('task_grocery', 3) = {decode_value('task_grocery', 3)}")
    print(f"decode_value('task_grocery', -9) = {decode_value('task_grocery', -9)}")
    print(f"decode_value('age', 45) = {decode_value('age', 45)}")
    
    print("\n=== Example 4: Access variable metadata ===")
    var_info = mapping['variables']['att_warm_relationship']
    print(f"Variable: att_warm_relationship")
    print(f"  Category: {var_info['category']}")
    print(f"  Description: {var_info['description']}")
    print(f"  Uses scale: {var_info.get('uses_scale', 'N/A')}")
    print(f"  Note: {var_info.get('note', 'N/A')}")

