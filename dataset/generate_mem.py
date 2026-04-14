import struct
import math
import pandas as pd

# Convert a standard float to BFloat16 Hex (top 16 bits of standard 32-bit float)
def float_to_bf16_hex(f):
    if math.isnan(f) or f == 0.0:
        return "0000"
    packed = struct.pack('>f', f) 
    return f"{packed[0]:02X}{packed[1]:02X}"

def excel_to_mem(excel_file, sheet_name, output_mem):
    print(f"Reading tab '{sheet_name}' from {excel_file}...")
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        
        values = []
        for index, row in df.iterrows():
            # BUGFIX: row.values[1:] forces it to skip the 'W1_row' or 'B1_index' numbering column entirely!
            for val in row.values[1:]:
                if pd.notna(val):  
                    try:
                        num = float(val)
                        values.append(num)
                    except ValueError:
                        pass # Safely ignore text headers
        
        # Pad with zeros if the total count isn't a multiple of 4
        while len(values) % 4 != 0:
            values.append(0.0)

        with open(output_mem, 'w') as f:
            # Pack 4 BFloat16 values into one 64-bit line 
            for i in range(0, len(values), 4):
                hex_str = (float_to_bf16_hex(values[i+3]) + 
                           float_to_bf16_hex(values[i+2]) + 
                           float_to_bf16_hex(values[i+1]) + 
                           float_to_bf16_hex(values[i]))
                f.write(f"{hex_str}\n")
                
        print(f"  -> Success! Generated {output_mem} ({len(values)} actual weights packed into {len(values)//4} lines)")
    except Exception as e:
        print(f"  [X] Error processing sheet '{sheet_name}': {e}")

if __name__ == "__main__":
    print("================================================")
    print("  EXCEL TO FPGA MEMORY CONVERTER (Fixed Alignment)")
    print("================================================\n")

    print("--- Features ---")
    excel_to_mem("instance_test.xlsx", "test_inputs", "feature_data.mem")
    
    print("\n--- Weights ---")
    excel_to_mem("instance_params.xlsx", "W1", "weight_L1.mem")
    excel_to_mem("instance_params.xlsx", "W2", "weight_L2.mem")
    excel_to_mem("instance_params.xlsx", "W3", "weight_L3.mem")
    
    print("\n--- Biases ---")
    excel_to_mem("instance_params.xlsx", "B1", "bias_L1.mem")
    excel_to_mem("instance_params.xlsx", "B2", "bias_L2.mem")
    excel_to_mem("instance_params.xlsx", "B3", "bias_L3.mem")