import tkinter as tk
from tkinter import ttk
import pickle
import torch
import numpy as np
from PIL import Image, ImageTk
import os
from threading import Thread


SEARCH_DIRS = ['.']  #'data', 'tmp',

class LoadingDialog:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Loading Dataset")
        self.top.transient(parent)
        # Do not call grab_set here; wait until window is mapped
        self.top.bind("<Map>", self._on_map)

        # Center the dialog over the parent window
        w = 300
        h = 200
        parent.update_idletasks()  # Ensure geometry info is up-to-date
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.top.geometry(f'{w}x{h}+{x}+{y}')

        self.frame = ttk.Frame(self.top, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Animated logo display
        from PIL import Image, ImageTk
        import os
        self.logo_paths = [
            os.path.join("assets", f"logo_a{i}.png") for i in range(4)
        ]
        self.logos = []
        for path in self.logo_paths:
            try:
                img = Image.open(path)
                img = img.resize((80, 80), Image.Resampling.LANCZOS)
                self.logos.append(ImageTk.PhotoImage(img))
            except Exception:
                self.logos.append(None)
        self.logo_idx = 0
        # Create a row frame for logo and progress bar
        logo_row = ttk.Frame(self.frame)
        logo_row.grid(row=0, column=0, pady=(0, 5), sticky="ew")
        logo_row.columnconfigure(0, weight=0)
        logo_row.columnconfigure(1, weight=1)
        self.logo_label = ttk.Label(logo_row)
        self.logo_label.grid(row=0, column=0, padx=(0, 10), sticky="w")
        def animate_logo():
            if self.logos[self.logo_idx]:
                self.logo_label.config(image=self.logos[self.logo_idx])
            self.logo_idx = (self.logo_idx + 1) % len(self.logos)
            self.top.after(40, animate_logo)
        animate_logo()
        self.progress = ttk.Progressbar(logo_row, mode='indeterminate', length=120)
        self.progress.grid(row=0, column=1, sticky="ew")
        self.progress.start(10)

        self.label = ttk.Label(self.frame, text="Loading dataset...")
        self.label.grid(row=1, column=0, pady=5)

    def _on_map(self, event=None):
        self.top.grab_set()
        self.top.unbind("<Map>")

    def destroy(self):
        self.top.destroy()

class DatasetSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Analyser")
        
        # Set window icon (cross-platform)
        import sys
        try:
            if sys.platform.startswith('win'):
                self.root.iconbitmap("./assets/CLOUDCELL-32x32.ico")
            else:
                icon_img = tk.PhotoImage(file="./assets/CLOUDCELL-32x32-0.png")
                self.root.wm_iconphoto(True, icon_img)
        except Exception as e:
            print(f"Warning: Could not load application icon: {e}")
        
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.minsize(600, 400)
        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky="nsew")
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

        ttk.Label(self.frame, text="Select a dataset to view:").grid(row=0, column=0, pady=10, sticky="w")

        # Scan for available datasets
        self.datasets = self.find_datasets()

        # Listbox frame for full-width expansion
        listbox_frame = tk.Frame(self.frame)
        listbox_frame.grid(row=1, column=0, sticky="nsew")
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)
        listbox_frame.grid_propagate(True)
        listbox_frame.update_idletasks()

        self.listbox = tk.Listbox(listbox_frame, borderwidth=1, relief="solid")
        self.listbox.grid(row=0, column=0, sticky="nsew")
        # Optional: Add a vertical scrollbar if many datasets
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scrollbar.set)

        for dataset in self.datasets:
            rel_path = os.path.relpath(dataset, os.getcwd())
            self.listbox.insert(tk.END, rel_path)

        if self.datasets:
            self.listbox.selection_set(0)

        # Enable double-click to open dataset (defer to avoid grab failed error)
        self.listbox.bind('<Double-Button-1>', lambda e: self.root.after_idle(self.open_dataset))

        ttk.Button(self.frame, text="Open Dataset", command=self.open_dataset).grid(row=2, column=0, pady=10, sticky="ew")
    
    def find_datasets(self):
        # Search recursively in all subfolders within 'data' and './tmp' for .pkl files
        search_dirs = SEARCH_DIRS
        datasets = []
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, _, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith('.pkl'):
                            datasets.append(os.path.join(root, file))
        # Remove duplicates, sort
        datasets = sorted(set(datasets))
        return datasets

    
    def open_dataset(self):
        selection = self.listbox.curselection()
        if not selection:
            return
        
        dataset_path = self.datasets[selection[0]]
        
        # Show loading dialog
        loading_dialog = LoadingDialog(self.root)
        self.root.update()
        
        # Load dataset in a separate thread
        def load_dataset():
            try:
                with open(dataset_path, 'rb') as f:
                    data = pickle.load(f)
                self.root.after(0, lambda: self.show_viewer(data, dataset_path))
            finally:
                self.root.after(0, loading_dialog.destroy)
        
        Thread(target=load_dataset, daemon=True).start()
    
    def show_viewer(self, data, dataset_path):
        # Hide selector frame
        self.frame.grid_remove()
        
        # Show viewer
        DatasetViewer(self.root, data['features'], data['labels'], dataset_path)

class DatasetViewer:
    def __init__(self, root, features, labels, dataset_path):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        self.root = root
        self.features = features
        self.labels = labels
        self.dataset_path = dataset_path
        self.current_idx = 0
        self.colormap_var = tk.StringVar(value="Viridis")

        # Top-level frame to hold everything
        self.top_frame = ttk.Frame(root)
        self.top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Image canvas (top, centered)
        self.canvas = tk.Canvas(self.top_frame, width=280, height=280, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, pady=8)

        # Add View menu for colormap selection
        self.create_menu_bar_with_view()

        # Main frame (rest of UI)
        self.main_frame = ttk.Frame(self.top_frame)
        self.main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.top_frame.grid_rowconfigure(1, weight=1)
        self.top_frame.grid_columnconfigure(0, weight=1)

        # --- rest of original __init__ continues below ---
        self.root = root
        self.features = features
        self.labels = labels
        self.dataset_path = dataset_path
        self.current_idx = 0
        
        # Set window icon (cross-platform)
        import sys
        try:
            if sys.platform.startswith('win'):
                self.root.iconbitmap("./assets/CLOUDCELL-32x32.ico")
            else:
                icon_img = tk.PhotoImage(file="./assets/CLOUDCELL-32x32-0.png")
                self.root.wm_iconphoto(True, icon_img)
        except Exception as e:
            print(f"Warning: Could not load application icon: {e}")
        
        # Default shape configuration
        self.feature_shape = None
        self.auto_detect_shape = True
        
        # Configure window to be resizable
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # Main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main frame columns and rows to be resizable
        self.main_frame.columnconfigure(0, weight=1)
        
        # Create menu bar

        
        # Image display
        self.canvas = tk.Canvas(self.main_frame, width=280, height=280, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Navigation frame
        nav_frame = ttk.Frame(self.main_frame)
        nav_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_sample).grid(row=0, column=0, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_sample).grid(row=0, column=1, padx=5)
        
        # Sample slider
        slider_frame = ttk.Frame(self.main_frame)
        slider_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Configure slider frame to expand with main frame
        slider_frame.columnconfigure(1, weight=1)
        
        ttk.Label(slider_frame, text="Sample:").grid(row=0, column=0, padx=5)
        self.sample_slider = ttk.Scale(slider_frame, orient=tk.HORIZONTAL, 
                                  from_=0, to=len(self.labels)-1,
                                  command=self.on_slider_change)
        self.sample_slider.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        self.sample_slider.set(self.current_idx)
        
        # Bind mouse wheel events to the slider for scrolling
        self.sample_slider.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows and macOS
        self.sample_slider.bind("<Button-4>", self.on_mouse_wheel)   # Linux scroll up
        self.sample_slider.bind("<Button-5>", self.on_mouse_wheel)   # Linux scroll down
        
        # Info frame
        info_frame = ttk.Frame(self.main_frame)
        info_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Configure info frame to expand with window
        info_frame.columnconfigure(1, weight=1)
        info_frame.columnconfigure(2, weight=1)
        
        ttk.Label(info_frame, text="Sample Index:").grid(row=0, column=0, padx=5)
        self.index_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.index_var, width=10).grid(row=0, column=1, padx=5)
        ttk.Button(info_frame, text="Go", command=self.go_to_index).grid(row=0, column=2, padx=5, pady=(0, 12))
        
        # Label display (use Text widget for highlighting)
        ttk.Label(info_frame, text="Label:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.label_text = tk.Text(info_frame, height=1, width=24, font=("TkDefaultFont", 10), borderwidth=0, highlightthickness=0)
        self.label_text.grid(row=1, column=1, columnspan=2, padx=5, sticky=tk.W+tk.E)
        self.label_text.config(state=tk.DISABLED)
        self.label_text.tag_configure('yellow', background='yellow')
        # Label for ASCII character name
        ttk.Label(info_frame, text="Char Name:").grid(row=2, column=0, padx=5, sticky=tk.W)
        self.char_name_var = tk.StringVar()
        self.char_name_label = ttk.Label(info_frame, textvariable=self.char_name_var, font=("TkDefaultFont", 10, "italic"))
        self.char_name_label.grid(row=2, column=1, columnspan=2, padx=5, sticky=tk.W)
        
        # File size display at the top
        try:
            file_size = os.path.getsize(self.dataset_path)
            file_size_str = f"File size: {file_size:,} bytes"
        except Exception:
            file_size_str = "File size: (unknown)"
        self.file_size_label = ttk.Label(info_frame, text=file_size_str, foreground="gray")
        self.file_size_label.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(0, 2))

        # Features as text display (move to next row)
        self.features_text_var = tk.StringVar()
        ttk.Label(info_frame, text="Features as Text:").grid(row=4, column=0, padx=5, sticky=tk.W)
        
        # Use a text widget for features text to allow scrolling and selection
        features_frame = ttk.Frame(info_frame)
        features_frame.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Configure features frame to expand with info frame
        features_frame.columnconfigure(0, weight=1)
        features_frame.rowconfigure(0, weight=1)
        
        # Create a text widget with scrollbar in a frame
        self.features_text = tk.Text(features_frame, height=6, wrap=tk.WORD)
        features_scrollbar = ttk.Scrollbar(features_frame, orient='vertical', command=self.features_text.yview)
        self.features_text.grid(row=0, column=0, sticky='nsew')
        features_scrollbar.grid(row=0, column=1, sticky='ns')
        features_frame.columnconfigure(0, weight=1)
        features_frame.rowconfigure(0, weight=1)
        self.features_text.config(yscrollcommand=features_scrollbar.set)
        # Make sure the scrollbar is always visible and tightly coupled
        self.features_text['yscrollcommand'] = features_scrollbar.set
        features_scrollbar['command'] = self.features_text.yview
        
        # Total samples
        ttk.Label(info_frame, text=f"Total Samples: {len(self.labels)}").grid(row=3, column=0, columnspan=3, pady=5)
        
        # --- Hex Viewer Panel ---
        self.main_frame.rowconfigure(4, weight=1)  # Make the hex viewer row expandable
        hex_frame = ttk.Frame(self.main_frame)
        hex_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.N, tk.S, tk.E, tk.W), pady=(10, 0))
        hex_frame.columnconfigure(0, weight=1)
        hex_frame.rowconfigure(1, weight=1)
        hex_label = ttk.Label(hex_frame, text="Sample Bytes (Hex View):")
        hex_label.pack(anchor=tk.W)
        self.hex_text = tk.Text(hex_frame, height=10, font=("Courier", 10), wrap="none")
        self.hex_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hex_scrollbar = ttk.Scrollbar(hex_frame, orient="vertical", command=self.hex_text.yview)
        self.hex_text.config(yscrollcommand=hex_scrollbar.set)
        hex_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.hex_text.config(state=tk.DISABLED)

        # Display first sample (must come after all widgets are created)
        self.show_current_sample()

        # Bind keyboard shortcuts
        root.bind('<Left>', lambda e: self.prev_sample())
        root.bind('<Right>', lambda e: self.next_sample())
        root.bind('<Return>', lambda e: self.go_to_index())
    
    def create_menu_bar_with_view(self):
        # Create menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        
        # Add file menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Different Dataset", command=self.open_different_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Add View menu for colormap selection
        view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Viridis", command=lambda: self.set_colormap_and_update('Viridis'))
        view_menu.add_command(label="Magenta", command=lambda: self.set_colormap_and_update('Magenta'))
        view_menu.add_command(label="Greyscale", command=lambda: self.set_colormap_and_update('Greyscale'))

        # Create Dataset menu
        dataset_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Dataset", menu=dataset_menu)
        
        # Add shape configuration submenu
        shape_menu = tk.Menu(dataset_menu, tearoff=0)
        dataset_menu.add_cascade(label="Shape", menu=shape_menu)
        shape_menu.add_command(label="Auto Shape", command=lambda: self.set_shape("auto"))
        shape_menu.add_separator()
        shape_menu.add_command(label="Custom Shape...", command=self.set_custom_shape)
        dataset_menu.add_command(label="Dataset Info", command=self.show_dataset_info)

        # Add Help menu with ASCII Reference
        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="ASCII Reference", command=self.show_ascii_reference)
        help_menu.add_command(label="DOS Charset (CP437)", command=self.show_dos_cp437)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about_dialog)

    def set_colormap_and_update(self, cmap_name):
        self.colormap_var.set(cmap_name)
        self.show_current_sample()
    
    def create_menu_bar(self):
        # Create menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        
        # Add file menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Different Dataset", command=self.open_different_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Create Dataset menu
        dataset_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Dataset", menu=dataset_menu)
        
        # Add shape configuration submenu
        shape_menu = tk.Menu(dataset_menu, tearoff=0)
        dataset_menu.add_cascade(label="Shape", menu=shape_menu)
        shape_menu.add_command(label="Auto Shape", command=lambda: self.set_shape("auto"))
        shape_menu.add_separator()
        shape_menu.add_command(label="Custom Shape...", command=self.set_custom_shape)
        dataset_menu.add_command(label="Dataset Info", command=self.show_dataset_info)

        # Add Help menu with ASCII Reference
        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="ASCII Reference", command=self.show_ascii_reference)
        help_menu.add_command(label="DOS Charset (CP437)", command=self.show_dos_cp437)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about_dialog)


        
    
    def set_shape(self, shape):
        if shape == "auto":
            self.auto_detect_shape = True
            self.feature_shape = None
        else:
            self.auto_detect_shape = False
            self.feature_shape = shape
        
        # Refresh the current image with the new shape
        self.show_current_sample()
    
    def set_custom_shape(self):
        # Create a dialog to get custom shape
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Custom Shape")
        dialog.geometry("300x250")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Frame for inputs
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Height and width inputs
        ttk.Label(frame, text="Height:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        height_var = tk.StringVar()
        height_entry = ttk.Entry(frame, textvariable=height_var, width=10)
        height_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame, text="Width:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        width_var = tk.StringVar()
        width_entry = ttk.Entry(frame, textvariable=width_var, width=10)
        width_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Feature size display
        feature_size = self.features[0].numel()
        ttk.Label(frame, text=f"Total Features: {feature_size}").grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # Error message display
        error_var = tk.StringVar()
        error_label = ttk.Label(frame, textvariable=error_var, foreground="red")
        error_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        def apply_shape():
            try:
                height = int(height_var.get())
                width = int(width_var.get())
                
                if height * width != feature_size:
                    error_var.set(f"Error: Height × Width must equal {feature_size}")
                    return
                
                self.set_shape((height, width))
                dialog.destroy()
            except ValueError:
                error_var.set("Error: Please enter valid numbers")
        
        ttk.Button(button_frame, text="Apply", command=apply_shape).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Set focus to height entry
        height_entry.focus_set()
    
    def show_dataset_info(self):
        import threading
        from tkinter import messagebox
        
        # Show progress dialog using LoadingDialog
        progress_dialog = LoadingDialog(self.root)
        progress_dialog.label.config(text="Calculating statistics...")
        self.root.update()

        def calc_and_show():
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from io import BytesIO
            import time
            import os
            # Compute min and max
            all_min = None
            all_max = None
            all_values = None
            try:
                if isinstance(self.features, torch.Tensor):
                    all_min = float(self.features.min().item())
                    all_max = float(self.features.max().item())
                    all_values = self.features.cpu().numpy().flatten()
                else:
                    all_min = float(np.min(self.features))
                    all_max = float(np.max(self.features))
                    all_values = np.array(self.features).flatten()
            except Exception as e:
                all_min = 'Error'
                all_max = 'Error'
                all_values = None

            # File size display
            try:
                file_size = os.path.getsize(self.dataset_path)
                file_size_str = f"File size: {file_size:,} bytes"
            except Exception:
                file_size_str = "File size: (unknown)"

            info_text = f"""
{file_size_str}
Dataset Path: {self.dataset_path}
Number of Samples: {len(self.labels)}
Feature Size: {self.features[0].numel()}
Current Shape: {self.feature_shape if not self.auto_detect_shape else 'Auto-detected'}
Min Pixel Value (all samples): {all_min}
Max Pixel Value (all samples): {all_max}
"""

            # Create histogram
            hist_img = None
            if all_values is not None and all_min != 'Error' and all_max != 'Error':
                bin_edges = np.linspace(all_min, all_max, 101)  # 100 bins
                fig, ax = plt.subplots(figsize=(4, 2.2), dpi=100)
                ax.hist(all_values, bins=bin_edges, color='skyblue', edgecolor='black')
                ax.set_title('Histogram of Pixel Values')
                ax.set_xlabel('Value')
                ax.set_ylabel('Count')
                plt.tight_layout()
                buf = BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                hist_img = Image.open(buf)

            # Now, close progress dialog and show info dialog
            progress_dialog.destroy()

            # Show info dialog (on main thread)
            def show_info():
                dialog = tk.Toplevel(self.root)
                dialog.title("Dataset Information")
                dialog.geometry("520x450")
                dialog.transient(self.root)
                dialog.grab_set()

                frame = ttk.Frame(dialog, padding="10")
                frame.pack(fill=tk.BOTH, expand=True)

                # Single text widget for info
                text_frame = ttk.Frame(frame)
                text_frame.pack(fill=tk.BOTH, expand=True)
                scrollbar = ttk.Scrollbar(text_frame)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, height=8)
                text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.config(command=text_widget.yview)
                text_widget.insert(tk.END, info_text)
                text_widget.config(state=tk.DISABLED)

                # Show histogram image if available
                if hist_img is not None:
                    self.hist_imgtk = ImageTk.PhotoImage(hist_img)
                    img_label = ttk.Label(frame, image=self.hist_imgtk)
                    img_label.pack(pady=(5, 0))

                ttk.Button(frame, text="Close", command=dialog.destroy).pack(pady=10)

            self.root.after(0, progress_dialog.destroy)
            self.root.after(0, show_info)

        # Run stats calculation in a background thread
        thread = threading.Thread(target=calc_and_show)
        thread.start()
        # The function returns immediately, the info dialog will appear when ready
    
    def show_about_dialog(self):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        import os
        dialog = tk.Toplevel(self.root)
        dialog.title("About CloudCell Dataset Viewer")
        dialog.geometry("420x420")
        dialog.transient(self.root)
        dialog.grab_set()
        # Icon (top left)
        try:
            icon_img = Image.open(os.path.join("assets", "CLOUDCELL-32x32-0.png"))
            icon_imgtk = ImageTk.PhotoImage(icon_img)
        except Exception:
            icon_imgtk = None
        # Logo
        # --- Animated logo for about dialog ---
        logo_paths = [os.path.join("assets", f"logo_a{i}.png") for i in range(4)]
        logo_imgs = []
        logo_imgtk_list = []
        max_width = 200
        for path in logo_paths:
            try:
                img = Image.open(path)
                if img.width > max_width:
                    scale = max_width / img.width
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                logo_imgs.append(img)
                logo_imgtk_list.append(ImageTk.PhotoImage(img))
            except Exception:
                logo_imgs.append(None)
                logo_imgtk_list.append(None)
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        # Animated logo label
        logo_idx = 3  # start at a3
        logo_label = ttk.Label(main_frame)
        if logo_imgtk_list[3]:
            logo_label.config(image=logo_imgtk_list[3])
            logo_label.image = logo_imgtk_list[3]
        logo_label.pack(side=tk.TOP, pady=(0,10))
        # Animation state
        logo_animating = {'job': None, 'idx': 0}
        rotating = {'active': False, 'job': None}
        rotated_imgtk = {'imgtk': None}
        import random
        def animate_logo():
            # cycles a0, a1, a2, a3
            if rotating['active']:
                return  # don't run hover animation if rotating
            logo_animating['idx'] = (logo_animating['idx'] + 1) % 4
            idx = logo_animating['idx']
            if logo_imgtk_list[idx]:
                logo_label.config(image=logo_imgtk_list[idx])
                logo_label.image = logo_imgtk_list[idx]
            logo_animating['job'] = dialog.after(40, animate_logo)
        def on_logo_enter(event):
            if rotating['active']:
                return
            logo_animating['idx'] = 0
            animate_logo()
        def on_logo_leave(event):
            if rotating['active']:
                return
            # stop animation and show a3
            if logo_animating['job']:
                dialog.after_cancel(logo_animating['job'])
                logo_animating['job'] = None
            if logo_imgtk_list[3]:
                logo_label.config(image=logo_imgtk_list[3])
                logo_label.image = logo_imgtk_list[3]
        def animate_rotation():
            # rotate a3 by a random angle and display
            angle = random.uniform(0, 360)
            base_img = logo_imgs[3]
            if base_img:
                rotated = base_img.rotate(angle, resample=Image.BICUBIC, expand=True)
                # Resize to fit max_width if needed
                if rotated.width > max_width:
                    scale = max_width / rotated.width
                    new_size = (int(rotated.width * scale), int(rotated.height * scale))
                    rotated = rotated.resize(new_size, Image.Resampling.LANCZOS)
                rotated_imgtk['imgtk'] = ImageTk.PhotoImage(rotated)
                logo_label.config(image=rotated_imgtk['imgtk'])
                logo_label.image = rotated_imgtk['imgtk']
            rotating['job'] = dialog.after(40, animate_rotation)
        def on_logo_click(event):
            if not rotating['active']:
                # Start rotating
                rotating['active'] = True
                # Stop hover animation if running
                if logo_animating['job']:
                    dialog.after_cancel(logo_animating['job'])
                    logo_animating['job'] = None
                animate_rotation()
            else:
                # Stop rotating
                rotating['active'] = False
                if rotating['job']:
                    dialog.after_cancel(rotating['job'])
                    rotating['job'] = None
                # Reset to a3
                if logo_imgtk_list[3]:
                    logo_label.config(image=logo_imgtk_list[3])
                    logo_label.image = logo_imgtk_list[3]
        def on_logo_double_click(event):
            if not rotating['active']:
                return  # Only allow while spinning
            # Pause rotation
            if rotating['job']:
                dialog.after_cancel(rotating['job'])
                rotating['job'] = None
            base_img = logo_imgs[3]
            orig_size = base_img.size if base_img else (100, 100)
            min_scale = 0.05
            steps = 12
            duration = 350  # ms for shrink or grow
            step_delay = duration // steps
            scales_down = [1 - (i / steps) * (1 - min_scale) for i in range(steps + 1)]
            scales_up = [min_scale + (i / steps) * (1 - min_scale) for i in range(steps + 1)]
            def show_scaled(scale):
                if base_img:
                    scaled = base_img.resize((max(1, int(orig_size[0]*scale)), max(1, int(orig_size[1]*scale))), Image.Resampling.LANCZOS)
                    rotated_imgtk['imgtk'] = ImageTk.PhotoImage(scaled)
                    logo_label.config(image=rotated_imgtk['imgtk'])
                    logo_label.image = rotated_imgtk['imgtk']
            def animate_shrink(idx=0):
                if idx < len(scales_down):
                    show_scaled(scales_down[idx])
                    dialog.after(step_delay, animate_shrink, idx+1)
                else:
                    dialog.after(80, animate_grow, 0)
            def animate_grow(idx=0):
                if idx < len(scales_up):
                    show_scaled(scales_up[idx])
                    dialog.after(step_delay, animate_grow, idx+1)
                else:
                    # Resume rotation if still active
                    if rotating['active']:
                        animate_rotation()
            animate_shrink(0)
        logo_label.bind('<Enter>', on_logo_enter)
        logo_label.bind('<Leave>', on_logo_leave)
        logo_label.bind('<Button-1>', on_logo_click)
        logo_label.bind('<Double-Button-1>', on_logo_double_click)

        # Title and version
        title_label = ttk.Label(main_frame, text="CloudCell Dataset Viewer", font=("Arial", 14, "bold"))
        title_label.pack(side=tk.TOP, anchor=tk.CENTER)
        # About text
        about_text = (
            "A user-friendly dataset viewer for CloudCell projects.\n"
            "Supports .pkl datasets, image and feature navigation, and quick ASCII/codepage references.\n"
            "\nCopyright © 2025 CloudCell. All rights reserved."
        )
        text_label = ttk.Label(main_frame, text=about_text, justify=tk.CENTER, wraplength=380)
        text_label.pack(side=tk.TOP, pady=(10,5))
        ttk.Button(main_frame, text="Close", command=dialog.destroy).pack(side=tk.BOTTOM, pady=10)

    def show_dos_cp437(self):
        # DOS Code Page 437 table for 128-255
        import tkinter as tk
        from tkinter import ttk
        dialog = tk.Toplevel(self.root)
        dialog.title("DOS Charset (CP437) 128–255")
        dialog.geometry("520x600")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(frame, wrap=tk.NONE, height=32, width=60, font=("Courier", 10))
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)

        text.insert(tk.END, f"{'Dec':>3}  {'Hex':>4}  {'Char':^7}  {'CP437 Description'}\n")
        text.insert(tk.END, f"{'-'*3}  {'-'*4}  {'-'*7}  {'-'*20}\n")
        # CP437 descriptions for 128-255 (abbreviated for brevity; full can be added as needed)
        cp437_desc = [
            'Ç','ü','é','â','ä','à','å','ç','ê','ë','è','ï','î','ì','Ä','Å',
            'É','æ','Æ','ô','ö','ò','û','ù','ÿ','Ö','Ü','¢','£','¥','₧','ƒ',
            'á','í','ó','ú','ñ','Ñ','ª','º','¿','⌐','¬','½','¼','¡','«','»',
            '░','▒','▓','│','┤','Á','Â','À','©','╣','║','╗','╝','¢','¥','┐',
            '└','┴','┬','├','─','┼','ã','Ã','╚','╔','╩','╦','╠','═','╬','¤',
            'ð','Ð','Ê','Ë','È','ı','Í','Î','Ï','┘','┌','█','▄','¦','Ì','▀',
            'Ó','ß','Ô','Ò','õ','Õ','µ','þ','Þ','Ú','Û','Ù','ý','Ý','¯','´',
            '≡','±','‗','¾','¶','§','÷','¸','°','¨','·','¹','³','²','■',' ' 
        ]
        cp437_names = [
            'Latin Capital Letter C with Cedilla', 'Latin Small Letter U with Diaeresis', 'Latin Small Letter E with Acute',
            'Latin Small Letter A with Circumflex', 'Latin Small Letter A with Diaeresis', 'Latin Small Letter A with Grave',
            'Latin Small Letter A with Ring Above', 'Latin Small Letter C with Cedilla', 'Latin Small Letter E with Circumflex',
            'Latin Small Letter E with Diaeresis', 'Latin Small Letter E with Grave', 'Latin Small Letter I with Diaeresis',
            'Latin Small Letter I with Circumflex', 'Latin Small Letter I with Grave', 'Latin Capital Letter A with Diaeresis',
            'Latin Capital Letter A with Ring Above', 'Latin Capital Letter E with Acute', 'Latin Small Letter AE',
            'Latin Capital Letter AE', 'Latin Small Letter O with Circumflex', 'Latin Small Letter O with Diaeresis',
            'Latin Small Letter O with Grave', 'Latin Small Letter U with Circumflex', 'Latin Small Letter U with Grave',
            'Latin Small Letter Y with Diaeresis', 'Latin Capital Letter O with Diaeresis', 'Latin Capital Letter U with Diaeresis',
            'Cent Sign', 'Pound Sign', 'Yen Sign', 'Peseta Sign', 'Function Sign', 'Latin Small Letter A with Acute',
            'Latin Small Letter I with Acute', 'Latin Small Letter O with Acute', 'Latin Small Letter U with Acute',
            'Latin Small Letter N with Tilde', 'Latin Capital Letter N with Tilde', 'Feminine Ordinal Indicator',
            'Masculine Ordinal Indicator', 'Inverted Question Mark', 'Reversed Not Sign', 'Not Sign', 'One Half',
            'One Quarter', 'Inverted Exclamation Mark', 'Left-Pointing Double Angle Quotation Mark',
            'Right-Pointing Double Angle Quotation Mark', 'Light Shade', 'Medium Shade', 'Dark Shade', 'Box Drawings Light Vertical',
            'Box Drawings Light Vertical and Left', 'Box Drawings Vertical Single and Left Double',
            'Box Drawings Vertical Double and Left Single', 'Box Drawings Down Double and Left Single',
            'Box Drawings Down Single and Left Double', 'Box Drawings Double Vertical and Right Single',
            'Box Drawings Double Vertical', 'Box Drawings Double Down and Left', 'Box Drawings Double Up and Left',
            'Box Drawings Up Double and Left Single', 'Box Drawings Up Single and Left Double', 'Box Drawings Light Down and Left',
            'Box Drawings Light Up and Right', 'Box Drawings Light Up and Horizontal', 'Box Drawings Light Down and Horizontal',
            'Box Drawings Light Vertical and Right', 'Box Drawings Light Horizontal', 'Box Drawings Light Vertical and Horizontal',
            'Box Drawings Vertical Single and Right Double', 'Box Drawings Vertical Double and Right Single',
            'Box Drawings Double Up and Right', 'Box Drawings Double Down and Right', 'Box Drawings Double Up and Horizontal',
            'Box Drawings Double Down and Horizontal', 'Box Drawings Double Vertical and Right', 'Box Drawings Double Horizontal',
            'Box Drawings Double Vertical and Horizontal', 'Box Drawings Up Single and Horizontal Double',
            'Box Drawings Down Single and Horizontal Double', 'Box Drawings Up Double and Horizontal Single',
            'Box Drawings Down Double and Horizontal Single', 'Box Drawings Up Double and Right Single',
            'Box Drawings Up Single and Right Double', 'Box Drawings Down Single and Right Double',
            'Box Drawings Down Double and Right Single', 'Box Drawings Vertical Double and Horizontal Single',
            'Box Drawings Vertical Single and Horizontal Double', 'Box Drawings Light Up and Left',
            'Box Drawings Light Down and Right', 'Full Block', 'Lower Half Block', 'Left Half Block', 'Right Half Block',
            'Upper Half Block', 'Greek Small Letter Alpha', 'Latin Small Letter Sharp S', 'Greek Capital Letter Gamma',
            'Greek Small Letter Pi', 'Greek Capital Letter Sigma', 'Greek Small Letter Sigma', 'Micro Sign',
            'Greek Small Letter Tau', 'Greek Capital Letter Phi', 'Greek Capital Letter Theta', 'Greek Capital Letter Omega',
            'Greek Small Letter Delta', 'Infinity', 'Greek Small Letter Phi', 'Greek Small Letter Epsilon', 'Intersection',
            'Identical To', 'Plus-Minus Sign', 'Greater-Than or Equal To', 'Less-Than or Equal To', 'Integral Top',
            'Integral Bottom', 'Division Sign', 'Almost Equal To', 'Degree Sign', 'Bullet Operator', 'Middle Dot',
            'Square Root', 'Superscript Latin Small Letter N', 'Superscript Two', 'Black Square', 'Non-Breaking Space'
        ]
        for i, code in enumerate(range(128, 256)):
            dec = f"{code:3}"
            hex_ = f"0x{code:02X}"
            try:
                c = bytes([code]).decode('cp437')
            except Exception:
                c = ''
            # Special handling for 255: display as space with description
            if code == 255:
                char = "' '"
                desc = 'Non-Breaking Space'
            else:
                char = f"'{c}'" if c.strip() else ''
                desc = cp437_names[i] if i < len(cp437_names) else ''
            text.insert(tk.END, f"{dec}  {hex_:>4}  {char:^7}  {desc}\n")
        text.config(state=tk.DISABLED)
        ttk.Button(frame, text="Close", command=dialog.destroy).pack(pady=10)

    def show_ascii_reference(self):
        import unicodedata
        dialog = tk.Toplevel(self.root)
        dialog.title("ASCII Reference Table")
        dialog.geometry("520x600")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Use a Text widget for scrollable table
        text = tk.Text(frame, wrap=tk.NONE, height=32, width=60, font=("Courier", 10))
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)

        # Table header
        text.insert(tk.END, f"{'Dec':>3}  {'Hex':>4}  {'Char':^7}  {'Name'}\n")
        text.insert(tk.END, f"{'-'*3}  {'-'*4}  {'-'*7}  {'-'*20}\n")
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
        for code in range(128):
            dec = f"{code:3}"
            hex_ = f"0x{code:02X}"
            if 32 <= code <= 126:
                char = f"'{chr(code)}'"
            else:
                char = ""
            if code < 32:
                mnemonic, explanation = ascii_mnemonics[code]
                name = f"{mnemonic}: {explanation}"
            elif code == 32:
                name = "SPACE: Space"
            elif code == 127:
                name = "DEL: Delete"
            else:
                try:
                    name = unicodedata.name(chr(code))
                except ValueError:
                    name = ''
            text.insert(tk.END, f"{dec}  {hex_:>4}  {char:^7}  {name}\n")

        # Add Latin-1/ISO-8859-1 section
        text.insert(tk.END, "\nLatin-1 (ISO-8859-1) Codes 128–255\n")
        text.insert(tk.END, f"{'Dec':>3}  {'Hex':>4}  {'Char':^7}  {'Unicode Name'}\n")
        text.insert(tk.END, f"{'-'*3}  {'-'*4}  {'-'*7}  {'-'*20}\n")
        control_explanations = {
            0: "Null",
            1: "Start of Heading",
            2: "Start of Text",
            3: "End of Text",
            4: "End of Transmission",
            5: "Enquiry",
            6: "Acknowledge",
            7: "Bell",
            8: "Backspace",
            9: "Horizontal Tab",
            10: "Line Feed",
            11: "Vertical Tab",
            12: "Form Feed",
            13: "Carriage Return",
            14: "Shift Out",
            15: "Shift In",
            16: "Data Link Escape",
            17: "Device Control 1",
            18: "Device Control 2",
            19: "Device Control 3",
            20: "Device Control 4",
            21: "Negative Acknowledge",
            22: "Synchronous Idle",
            23: "End of Transmission Block",
            24: "Cancel",
            25: "End of Medium",
            26: "Substitute",
            27: "Escape",
            28: "File Separator",
            29: "Group Separator",
            30: "Record Separator",
            31: "Unit Separator",
            127: "Delete"
        }
        for code in range(256):
            dec = f"{code:3}"
            hex_ = f"0x{code:02X}"
            char = ''
            name = ''
            if 32 <= code <= 126 or 160 <= code <= 255:
                c = bytes([code]).decode('latin-1')
                char = f"'{c}'"
                try:
                    name = unicodedata.name(c)
                except Exception:
                    name = ''
            else:
                # Control characters (C0: 0–31, 127; C1: 128–159)
                try:
                    name = unicodedata.name(chr(code))
                except Exception:
                    # Fallback for C1 controls (128–159) which may not have names in unicodedata
                    c1_names = [
                        'PADDING CHARACTER', 'HIGH OCTET PRESET', 'BREAK PERMITTED HERE', 'NO BREAK HERE',
                        'INDEX', 'NEXT LINE', 'START OF SELECTED AREA', 'END OF SELECTED AREA',
                        'CHARACTER TABULATION SET', 'CHARACTER TABULATION WITH JUSTIFICATION',
                        'LINE TABULATION SET', 'PARTIAL LINE FORWARD', 'PARTIAL LINE BACKWARD',
                        'REVERSE LINE FEED', 'SINGLE SHIFT TWO', 'SINGLE SHIFT THREE',
                        'DEVICE CONTROL STRING', 'PRIVATE USE ONE', 'PRIVATE USE TWO', 'SET TRANSMIT STATE',
                        'CANCEL CHARACTER', 'MESSAGE WAITING', 'START OF GUARDED AREA', 'END OF GUARDED AREA',
                        'START OF STRING', '', '', '', '', '', '', ''
                    ]
                    if 128 <= code <= 159:
                        idx = code - 128
                        name = f'C1 CONTROL: {c1_names[idx]}' if c1_names[idx] else 'C1 CONTROL'
                    elif code == 127:
                        name = 'DELETE'
                    else:
                        # C0 controls (0–31)
                        try:
                            name = unicodedata.name(chr(code))
                        except Exception:
                            name = 'CONTROL'
    def create_image(self, features):
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image, ImageTk
        from matplotlib import cm

        # Get the size of the features
        feature_size = features.numel()
        # Determine the appropriate dimensions for visualization
        if self.auto_detect_shape:
            if feature_size == 784:  # MNIST (28x28)
                img_size = (28, 28)
            elif feature_size == 128:  # 128 features
                img_size = (8, 16)
            else:
                # For other sizes, try to make a square-ish image
                side = int(np.sqrt(feature_size))
                if side * side == feature_size:
                    img_size = (side, side)
                else:
                    # Find the closest factors
                    for i in range(side, 0, -1):
                        if feature_size % i == 0:
                            img_size = (i, feature_size // i)
                            break
                    else:
                        # If no factors found, use a rectangular shape
                        img_size = (1, feature_size)
        else:
            # Use the user-specified shape
            img_size = self.feature_shape
        
        # Reshape and convert to numpy array
        img_array = features.reshape(img_size).numpy()
        # Normalize for colormap (float in 0-1)
        ptp = np.ptp(img_array)
        arr_norm = (img_array - img_array.min()) / (ptp if ptp > 0 else 1)
        # Select colormap
        cmap_name = self.colormap_var.get()
        if cmap_name == "Viridis":
            cmap = cm.viridis
        elif cmap_name == "Magenta":
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("magenta", ["black", "magenta", "white"])
        else:  # Greyscale
            cmap = cm.gray
        arr_rgb = (cmap(arr_norm)[..., :3] * 255).astype(np.uint8)  # RGB
        # Scale up the image for better visibility (maintain aspect ratio)
        scale_factor = 280 / max(img_size)
        scaled_size = (int(img_size[1] * scale_factor), int(img_size[0] * scale_factor))
        img = Image.fromarray(arr_rgb).resize(scaled_size, Image.Resampling.NEAREST)
        return ImageTk.PhotoImage(img)
    
    def show_current_sample(self):
        # Update index display and slider
        self.index_var.set(str(self.current_idx))
        self.sample_slider.set(self.current_idx)
        
        # Get current sample
        features = self.features[self.current_idx]
        label = self.labels[self.current_idx].item()
        
        # Create and display image
        self.current_image = self.create_image(features)
        self.canvas.delete("all")
        self.canvas.create_image(140, 140, image=self.current_image)
        
        # Update label display with yellow highlight for ASCII char (dot if nonprintable)
        import unicodedata
        ascii_names = {
            0: 'NUL (null)', 1: 'SOH (start of heading)', 2: 'STX (start of text)', 3: 'ETX (end of text)',
            4: 'EOT (end of transmission)', 5: 'ENQ (enquiry)', 6: 'ACK (acknowledge)', 7: 'BEL (bell)',
            8: 'BS (backspace)', 9: 'TAB (horizontal tab)', 10: 'LF (line feed)', 11: 'VT (vertical tab)',
            12: 'FF (form feed)', 13: 'CR (carriage return)', 14: 'SO (shift out)', 15: 'SI (shift in)',
            16: 'DLE (data link escape)', 17: 'DC1 (device control 1)', 18: 'DC2 (device control 2)', 19: 'DC3 (device control 3)',
            20: 'DC4 (device control 4)', 21: 'NAK (negative acknowledge)', 22: 'SYN (synchronous idle)', 23: 'ETB (end of trans. block)',
            24: 'CAN (cancel)', 25: 'EM (end of medium)', 26: 'SUB (substitute)', 27: 'ESC (escape)',
            28: 'FS (file separator)', 29: 'GS (group separator)', 30: 'RS (record separator)', 31: 'US (unit separator)',
            127: 'DEL (delete)'
        }
        if 32 <= label < 127:
            char_display = chr(label)
        else:
            char_display = '.'
        label_str = f"ASCII: {label} ('{char_display}')"
        self.label_text.config(state=tk.NORMAL)
        self.label_text.delete(1.0, tk.END)
        self.label_text.insert(tk.END, label_str)
        # Highlight the displayed char (or dot) in yellow
        idx = label_str.find(f"'{char_display}'")
        if idx != -1:
            start = f"1.{idx+1}"
            end = f"1.{idx+2}"
            self.label_text.tag_add('yellow', start, end)
        self.label_text.config(state=tk.DISABLED)
        # Set the ASCII character name label
        if label in ascii_names:
            self.char_name_var.set(ascii_names[label])
        else:
            try:
                self.char_name_var.set(unicodedata.name(chr(label)))
            except Exception:
                self.char_name_var.set('N/A')
        
        # Convert features to ASCII text
        feature_values = features.numpy()
        feature_text = ""
        
        # Try to interpret features as ASCII characters
        try:
            # Check if features are binary (0s and 1s)
            is_binary = np.all(np.isin(feature_values, [0, 1]))
            
            if is_binary:
                # Process as 8-bit sequences (bytes)
                feature_size = len(feature_values)
                bytes_data = []
                
                # Process every 8 bits as a byte
                for i in range(0, feature_size, 8):
                    if i + 8 <= feature_size:
                        # Convert 8 bits to a byte
                        byte_value = 0
                        for j in range(8):
                            byte_value = (byte_value << 1) | int(feature_values[i + j])
                        bytes_data.append(byte_value)
                
                # Convert bytes to ASCII characters (only printable range)
                printable_chars = []
                for byte in bytes_data:
                    if 32 <= byte <= 126:  # Printable ASCII range
                        printable_chars.append(chr(byte))
                
                if printable_chars:
                    feature_text = "".join(printable_chars)
                else:
                    # If no printable characters, show binary representation
                    feature_text = "Binary: " + " ".join([f"{byte:08b}" for byte in bytes_data])
            
            # If not binary or no bytes were processed, fall back to previous methods
            if not is_binary or not feature_text:
                if feature_values.max() <= 1.0:
                    # Scale to 0-255 range
                    ascii_values = (feature_values * 255).astype(np.uint8)
                    # Filter to printable ASCII range (32-126)
                    printable_indices = np.where((ascii_values >= 32) & (ascii_values <= 126))[0]
                    if len(printable_indices) > 0:
                        printable_chars = [chr(int(ascii_values[i])) for i in printable_indices]
                        feature_text = "".join(printable_chars)
                    else:
                        feature_text = "(No printable ASCII characters found)"
                else:
                    # For features already in numeric range, convert directly
                    ascii_values = feature_values.astype(np.uint8)
                    # Filter to printable ASCII range (32-126)
                    printable_indices = np.where((ascii_values >= 32) & (ascii_values <= 126))[0]
                    if len(printable_indices) > 0:
                        printable_chars = [chr(int(ascii_values[i])) for i in printable_indices]
                        feature_text = "".join(printable_chars)
                    else:
                        feature_text = "(No printable ASCII characters found)"
        except Exception as e:
            feature_text = f"Error converting to ASCII: {str(e)}"
        
        # Update features text display: show all bytes as characters, non-printable as · (dot), highlight in yellow
        self.features_text.config(state=tk.NORMAL)
        self.features_text.delete(1.0, tk.END)
        feature_bytes = None
        try:
            if feature_values.max() <= 1.0:
                arr = (feature_values * 255).astype(np.uint8)
            else:
                arr = feature_values.astype(np.uint8)
            feature_bytes = arr.tolist()
        except Exception:
            feature_bytes = []
        display_chars = []
        nonprintable_indices = []
        for idx, b in enumerate(feature_bytes):
            if 32 <= b < 127:
                display_chars.append(chr(b))
            else:
                display_chars.append('·')
                nonprintable_indices.append(idx)
        display_str = ''.join(display_chars)
        self.features_text.insert(tk.END, display_str)
        # Highlight non-printable characters in yellow
        self.features_text.tag_delete('nonprintable')
        self.features_text.tag_configure('nonprintable', background='yellow')
        for idx in nonprintable_indices:
            start = f"1.{idx}"
            end = f"1.{idx+1}"
            self.features_text.tag_add('nonprintable', start, end)
        self.features_text.config(state=tk.DISABLED)


        # --- Update hex viewer ---
        # Try to get raw bytes from features
        try:
            # If binary, use bytes_data from above; else, try to convert feature_values to bytes
            if 'bytes_data' in locals() and bytes_data:
                raw_bytes = bytes(bytes_data)
            else:
                # Try to scale or cast to bytes
                if feature_values.max() <= 1.0:
                    arr = (feature_values * 255).astype(np.uint8)
                else:
                    arr = feature_values.astype(np.uint8)
                raw_bytes = bytes(arr)
        except Exception:
            raw_bytes = b''

        self.hex_text.config(state=tk.NORMAL)
        self.hex_text.delete(1.0, tk.END)
        self.hex_text.tag_delete('nonprintable')
        self.hex_text.tag_configure('nonprintable', background='yellow')

        for line_no, offset in enumerate(range(0, len(raw_bytes), 16), start=1):
            chunk = raw_bytes[offset:offset+16]
            # Hex bytes with | after each 4 bytes
            hex_groups = []
            for i in range(0, 16, 4):
                group = ' '.join(f"{b:02X}" for b in chunk[i:i+4]) if i < len(chunk) else ''
                # Pad group with spaces if less than 4 bytes
                if len(chunk[i:i+4]) < 4:
                    group = group.ljust(3*4-1)
                hex_groups.append(group)
            hex_bytes = ' | '.join(hex_groups)
            # Pad the entire hex_bytes to match a full row
            hex_bytes = hex_bytes.ljust(16*3 + 3)

            # ASCII representation, pad with spaces for missing bytes
            ascii_bytes = ''
            for b in chunk:
                ascii_bytes += chr(b) if 32 <= b < 127 else '.'
            ascii_bytes = ascii_bytes.ljust(16)

            line = f"{offset:08X}  {hex_bytes}  {ascii_bytes}\n"
            self.hex_text.insert(tk.END, line)
            # Highlight nonprintable chars in ASCII column
            ascii_start = line.find(ascii_bytes)
            for i, b in enumerate(chunk):
                if not (32 <= b < 127):
                    tag_start = f"{line_no}.{ascii_start + i}"
                    tag_end = f"{line_no}.{ascii_start + i + 1}"
                    self.hex_text.tag_add('nonprintable', tag_start, tag_end)

        if len(raw_bytes) == 0:
            self.hex_text.insert(tk.END, '(No data)')
        self.hex_text.config(state=tk.DISABLED)
    
    def next_sample(self):
        if self.current_idx < len(self.labels) - 1:
            self.current_idx += 1
            self.show_current_sample()
    
    def prev_sample(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_sample()
    
    def go_to_index(self):
        try:
            idx = int(self.index_var.get())
            if 0 <= idx < len(self.labels):
                self.current_idx = idx
                self.show_current_sample()
        except ValueError:
            pass
    
    def on_slider_change(self, value):
        # Convert value to int and update current index
        try:
            new_idx = int(float(value))
            if new_idx != self.current_idx:
                self.current_idx = new_idx
                self.show_current_sample()
        except ValueError:
            pass
            
    def on_mouse_wheel(self, event):
        # Determine scroll direction and update index accordingly
        if event.num == 4 or event.delta > 0:  # Scroll up (Linux) or up (Windows/macOS)
            self.prev_sample()
        elif event.num == 5 or event.delta < 0:  # Scroll down (Linux) or down (Windows/macOS)
            self.next_sample()

    def open_different_dataset(self):
        # Clear the main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Remove the menu bar
        self.root.config(menu="")
        
        # Show the dataset selector again
        selector = DatasetSelector(self.root)
        selector.frame.grid()

def main():
    root = tk.Tk()
    root.withdraw()  # Hide window until geometry is set
    # Center the main window on the active monitor (multi-monitor aware)
    w = 800
    h = 900  # Height for hex viewer
    centered = False
    try:
        from screeninfo import get_monitors
        pointer_x = root.winfo_pointerx()
        pointer_y = root.winfo_pointery()
        monitors = get_monitors()
        monitor = monitors[0]
        for m in monitors:
            if (m.x <= pointer_x < m.x + m.width) and (m.y <= pointer_y < m.y + m.height):
                monitor = m
                break
        x = monitor.x + (monitor.width - w) // 2
        y = monitor.y + (monitor.height - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")
        centered = True
    except Exception:
        centered = False

    if not centered:
        try:
            from screeninfo import get_monitors
            pointer_x = pointer_y = None
            try:
                import tkinter
                pointer_x = root.winfo_pointerx()
                pointer_y = root.winfo_pointery()
            except Exception:
                pointer_x = pointer_y = None
            monitors = get_monitors()
            monitor = monitors[0]
            if pointer_x is not None and pointer_y is not None:
                for m in monitors:
                    if (m.x <= pointer_x < m.x + m.width) and (m.y <= pointer_y < m.y + m.height):
                        monitor = m
                        break
            x = monitor.x + (monitor.width - w) // 2
            y = monitor.y + (monitor.height - h) // 2
            root.geometry(f"{w}x{h}+{x}+{y}")
            centered = True
        except ImportError:
            centered = False
    if not centered:
        root.update_idletasks()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")
    root.deiconify()  # Show window after geometry is set
    app = DatasetSelector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
