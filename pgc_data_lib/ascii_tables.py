"""
ASCII Tables Library

This module provides functions for displaying ASCII and CP437 character tables
in a Tkinter text widget.
"""

import unicodedata

def populate_ascii_tables_widget(text_widget):
    """
    Populates a Tkinter text widget with ASCII and CP437 character tables.
    
    Args:
        text_widget: A Tkinter text widget (or scrolled text) to populate
    """
    # ASCII standard table (0-127)
    text_widget.insert("end", "ASCII Standard Characters Codes 0–127\n")
    text_widget.insert("end", f"{'Dec':>3}  {'Hex':>4}  {'Char':^7}  {'Name'}\n")
    text_widget.insert("end", f"{'-'*3}  {'-'*4}  {'-'*7}  {'-'*20}\n")
    
    # Mapping of ASCII control character mnemonics to explanations
    ascii_mnemonics = [
        ('NUL', 'Null'),
        ('SOH', 'Start of Heading'),
        ('STX', 'Start of Text'),
        ('ETX', 'End of Text'),
        ('EOT', 'End of Transmission'),
        ('ENQ', 'Enquiry'),
        ('ACK', 'Acknowledge'),
        ('BEL', 'Bell'),
        ('BS', 'Backspace'),
        ('TAB', 'Horizontal Tab'),
        ('LF', 'Line Feed'),
        ('VT', 'Vertical Tab'),
        ('FF', 'Form Feed'),
        ('CR', 'Carriage Return'),
        ('SO', 'Shift Out'),
        ('SI', 'Shift In'),
        ('DLE', 'Data Link Escape'),
        ('DC1', 'Device Control 1'),
        ('DC2', 'Device Control 2'),
        ('DC3', 'Device Control 3'),
        ('DC4', 'Device Control 4'),
        ('NAK', 'Negative Acknowledge'),
        ('SYN', 'Synchronous Idle'),
        ('ETB', 'End of Transmission Block'),
        ('CAN', 'Cancel'),
        ('EM', 'End of Medium'),
        ('SUB', 'Substitute'),
        ('ESC', 'Escape'),
        ('FS', 'File Separator'),
        ('GS', 'Group Separator'),
        ('RS', 'Record Separator'),
        ('US', 'Unit Separator')
    ]
    
    # Display ASCII standard characters
    for code in range(128):
        dec = f"{code:3}"
        hex_ = f"0x{code:02X}"
        
        if code < 32:  # Control characters
            char = f"'{ascii_mnemonics[code][0]}'"
            name = ascii_mnemonics[code][1]
        elif code == 127:  # DEL
            char = "'DEL'"
            name = "Delete"
        else:  # Printable characters
            char = f"'{chr(code)}'"
            try:
                name = unicodedata.name(chr(code))
            except ValueError:
                name = ''
        
        text_widget.insert("end", f"{dec}  {hex_:>4}  {char:^7}  {name}\n")
    
    # ASCII Extended Characters section (CP437)
    text_widget.insert("end", "\nASCII Extended Characters (CP437) Codes 128–255\n")
    text_widget.insert("end", f"{'Dec':>3}  {'Hex':>4}  {'Char':^7}  {'Character Description'}\n")
    text_widget.insert("end", f"{'-'*3}  {'-'*4}  {'-'*7}  {'-'*20}\n")
    
    # Define CP437 character descriptions
    cp437_descriptions = {
        128: 'Latin Capital Letter C with Cedilla',
        129: 'Latin Small Letter U with Diaeresis',
        130: 'Latin Small Letter E with Acute',
        131: 'Latin Small Letter A with Circumflex',
        132: 'Latin Small Letter A with Diaeresis',
        133: 'Latin Small Letter A with Grave',
        134: 'Latin Small Letter A with Ring Above',
        135: 'Latin Small Letter C with Cedilla',
        136: 'Latin Small Letter E with Circumflex',
        137: 'Latin Small Letter E with Diaeresis',
        138: 'Latin Small Letter E with Grave',
        139: 'Latin Small Letter I with Diaeresis',
        140: 'Latin Small Letter I with Circumflex',
        141: 'Latin Small Letter I with Grave',
        142: 'Latin Capital Letter A with Diaeresis',
        143: 'Latin Capital Letter A with Ring Above',
        144: 'Latin Capital Letter E with Acute',
        145: 'Latin Small Letter AE',
        146: 'Latin Capital Letter AE',
        147: 'Latin Small Letter O with Circumflex',
        148: 'Latin Small Letter O with Diaeresis',
        149: 'Latin Small Letter O with Grave',
        150: 'Latin Small Letter U with Circumflex',
        151: 'Latin Small Letter U with Grave',
        152: 'Latin Small Letter Y with Diaeresis',
        153: 'Latin Capital Letter O with Diaeresis',
        154: 'Latin Capital Letter U with Diaeresis',
        155: 'Cent Sign',
        156: 'Pound Sign',
        157: 'Yen Sign',
        158: 'Peseta Sign',
        159: 'Function Sign',
        160: 'Latin Small Letter A with Acute',
        161: 'Latin Small Letter I with Acute',
        162: 'Latin Small Letter O with Acute',
        163: 'Latin Small Letter U with Acute',
        164: 'Latin Small Letter N with Tilde',
        165: 'Latin Capital Letter N with Tilde',
        166: 'Feminine Ordinal Indicator',
        167: 'Masculine Ordinal Indicator',
        168: 'Inverted Question Mark',
        169: 'Reversed Not Sign',
        170: 'Not Sign',
        171: 'One Half',
        172: 'One Quarter',
        173: 'Inverted Exclamation Mark',
        174: 'Left-Pointing Double Angle Quotation Mark',
        175: 'Right-Pointing Double Angle Quotation Mark',
        176: 'Light Shade',
        177: 'Medium Shade',
        178: 'Dark Shade',
        179: 'Box Drawings Light Vertical',
        180: 'Box Drawings Light Vertical and Left',
        181: 'Box Drawings Vertical Single and Left Double',
        182: 'Box Drawings Vertical Double and Left Single',
        183: 'Box Drawings Down Double and Left Single',
        184: 'Box Drawings Down Single and Left Double',
        185: 'Box Drawings Double Vertical and Right Single',
        186: 'Box Drawings Double Vertical',
        187: 'Box Drawings Double Down and Left',
        188: 'Box Drawings Double Up and Left',
        189: 'Box Drawings Up Double and Left Single',
        190: 'Box Drawings Up Single and Left Double',
        191: 'Box Drawings Light Down and Left',
        192: 'Box Drawings Light Up and Right',
        193: 'Box Drawings Light Up and Horizontal',
        194: 'Box Drawings Light Down and Horizontal',
        195: 'Box Drawings Light Vertical and Right',
        196: 'Box Drawings Light Horizontal',
        197: 'Box Drawings Light Vertical and Horizontal',
        198: 'Box Drawings Vertical Single and Right Double',
        199: 'Box Drawings Vertical Double and Right Single',
        200: 'Box Drawings Double Up and Right',
        201: 'Box Drawings Double Down and Right',
        202: 'Box Drawings Double Up and Horizontal',
        203: 'Box Drawings Double Down and Horizontal',
        204: 'Box Drawings Double Vertical and Right',
        205: 'Box Drawings Double Horizontal',
        206: 'Box Drawings Double Vertical and Horizontal',
        207: 'Box Drawings Up Single and Horizontal Double',
        208: 'Box Drawings Down Single and Horizontal Double',
        209: 'Box Drawings Up Double and Horizontal Single',
        210: 'Box Drawings Down Double and Horizontal Single',
        211: 'Box Drawings Up Double and Right Single',
        212: 'Box Drawings Up Single and Right Double',
        213: 'Box Drawings Down Single and Right Double',
        214: 'Box Drawings Down Double and Right Single',
        215: 'Box Drawings Vertical Double and Horizontal Single',
        216: 'Box Drawings Vertical Single and Horizontal Double',
        217: 'Box Drawings Light Up and Left',
        218: 'Box Drawings Light Down and Right',
        219: 'Full Block',
        220: 'Lower Half Block',
        221: 'Left Half Block',
        222: 'Right Half Block',
        223: 'Upper Half Block',
        224: 'Greek Small Letter Alpha',
        225: 'Latin Small Letter Sharp S',
        226: 'Greek Capital Letter Gamma',
        227: 'Greek Small Letter Pi',
        228: 'Greek Capital Letter Sigma',
        229: 'Greek Small Letter Sigma',
        230: 'Micro Sign',
        231: 'Greek Small Letter Tau',
        232: 'Greek Capital Letter Phi',
        233: 'Greek Capital Letter Theta',
        234: 'Greek Capital Letter Omega',
        235: 'Greek Small Letter Delta',
        236: 'Infinity',
        237: 'Greek Small Letter Phi',
        238: 'Greek Small Letter Epsilon',
        239: 'Intersection',
        240: 'Identical To',
        241: 'Plus-Minus Sign',
        242: 'Greater-Than or Equal To',
        243: 'Less-Than or Equal To',
        244: 'Integral Top',
        245: 'Integral Bottom',
        246: 'Division Sign',
        247: 'Almost Equal To',
        248: 'Degree Sign',
        249: 'Bullet Operator',
        250: 'Middle Dot',
        251: 'Square Root',
        252: 'Superscript Latin Small Letter N',
        253: 'Superscript Two',
        254: 'Black Square',
        255: 'Non-Breaking Space'
    }
    
    # Display CP437 extended ASCII characters
    for code in range(128, 256):
        dec = f"{code:3}"
        hex_ = f"0x{code:02X}"
        
        # Get character description from our CP437 mapping
        name = cp437_descriptions.get(code, '')
        
        # Try to display the actual CP437 character if possible
        try:
            # Note: This is an approximation as modern terminals may not display CP437 correctly
            char = f"'\\x{code:02x}'"
        except Exception:
            char = f"'?'"
            
        text_widget.insert("end", f"{dec}  {hex_:>4}  {char:^7}  {name}\n")

def create_ascii_reference_dialog(parent_window):
    """
    Creates a dialog window displaying ASCII and CP437 character tables.
    
    Args:
        parent_window: The parent Tkinter window/widget
        
    Returns:
        The created dialog window
    """
    import tkinter as tk
    from tkinter import ttk
    
    dialog = tk.Toplevel(parent_window)
    dialog.title("ASCII Reference Table")
    dialog.geometry("520x600")
    dialog.transient(parent_window)
    dialog.grab_set()

    frame = ttk.Frame(dialog, padding="10")
    frame.pack(fill=tk.BOTH, expand=True)

    # Use a Text widget for scrollable table
    text = tk.Text(frame, wrap=tk.NONE, height=32, width=60, font=("Courier", 10))
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text.config(yscrollcommand=scrollbar.set)
    
    # Populate the text widget with ASCII tables
    populate_ascii_tables_widget(text)
    
    # Add a close button
    text.config(state=tk.DISABLED)
    ttk.Button(frame, text="Close", command=dialog.destroy).pack(pady=10)
    
    return dialog
