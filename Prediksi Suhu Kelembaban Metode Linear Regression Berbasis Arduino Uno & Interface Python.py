import serial
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# Konfigurasi serial
arduino = serial.Serial('COM6', 9600, timeout=1)

# Dataset awal
data = pd.DataFrame(columns=["Waktu", "Suhu", "Kelembapan"])
model_temp = LinearRegression()
model_hum = LinearRegression()

# Fungsi untuk memproses waktu ke format HH:MM:SS
def format_time(timestamp):
    return timestamp.strftime("%H:%M:%S")

# Fungsi untuk memperbarui data
def update_data():
    global data, model_temp, model_hum

    try:
        line = arduino.readline().decode().strip()
        if line:
            temp, hum = map(float, line.split(","))
            waktu = pd.Timestamp.now()

            # Tambahkan data ke dataset
            new_row = {"Waktu": waktu, "Suhu": temp, "Kelembapan": hum}
            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

            # Latih ulang model jika data cukup
            if len(data) > 10:
                # Menggunakan data waktu (dalam detik sejak awal pengambilan data)
                data["Elapsed"] = (data["Waktu"] - data["Waktu"].iloc[0]).dt.total_seconds()
                X = data[["Elapsed"]].values
                y_temp = data["Suhu"].values
                y_hum = data["Kelembapan"].values

                model_temp.fit(X, y_temp)
                model_hum.fit(X, y_hum)

                # Prediksi untuk 5 interval waktu ke depan (dalam detik)
                last_time = data["Elapsed"].iloc[-1]
                future_seconds = np.array([[last_time + i * 60] for i in range(1, 6)])  # Setiap 1 menit
                future_temp = model_temp.predict(future_seconds)
                future_hum = model_hum.predict(future_seconds)

                # Terapkan rata-rata bergerak untuk mengurangi fluktuasi
                future_temp = pd.Series(future_temp).rolling(window=2, min_periods=1).mean().values
                future_hum = pd.Series(future_hum).rolling(window=2, min_periods=1).mean().values

                return future_seconds.flatten(), future_temp, future_hum
    except Exception as e:
        print(f"Error: {e}")
    return None, None, None

# Fungsi untuk memperbarui plot, tabel prediksi, dan tabel terkini
def plot_real_time(frame):
    global data

    future_seconds, future_temp, future_hum = update_data()

    ax.clear()
    if len(data) > 0:
        # Format waktu
        formatted_time = data["Waktu"].dt.strftime("%H:%M:%S")
        ax.plot(formatted_time, data["Suhu"], label="Suhu Aktual", color="red")
        ax.plot(formatted_time, data["Kelembapan"], label="Kelembapan Aktual", color="blue")

        # Perbarui tabel terkini
        latest_row = data.iloc[-1]
        current_time = format_time(latest_row["Waktu"])
        current_temp = latest_row["Suhu"]
        current_hum = latest_row["Kelembapan"]
        current_table.item("current", values=(current_time, f"{current_temp:.2f}", f"{current_hum:.2f}"))

    ax.set_title("Prediksi Suhu dan Kelembapan Real-Time")
    ax.set_xlabel("Waktu (HH:MM:SS)")
    ax.set_ylabel("Nilai")
    ax.legend(loc="upper left")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    if future_seconds is not None:
        for i, (sec, temp, hum) in enumerate(zip(future_seconds, future_temp, future_hum)):
            future_time = data["Waktu"].iloc[0] + pd.Timedelta(seconds=sec)
            pred_table.item(f"I{i+1}", values=(format_time(future_time), f"{temp:.2f}", f"{hum:.2f}"))

# GUI menggunakan Tkinter
root = tk.Tk()
root.title("Prediksi Suhu dan Kelembapan Real-Time")

# Frame untuk plot
frame_plot = tk.Frame(root)
frame_plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Frame untuk tabel
frame_table = tk.Frame(root)
frame_table.pack(side=tk.RIGHT, fill=tk.Y)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=frame_plot)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Tabel prediksi
columns = ("Waktu", "Suhu", "Kelembapan")
pred_table = ttk.Treeview(frame_table, columns=columns, show="headings", height=5)
pred_table.heading("Waktu", text="Waktu")
pred_table.heading("Suhu", text="Suhu (°C)")
pred_table.heading("Kelembapan", text="Kelembapan (%)")
pred_table.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

for i in range(1, 6):
    pred_table.insert("", "end", iid=f"I{i}", values=("---", "---", "---"))

# Tabel terkini
current_table = ttk.Treeview(frame_table, columns=columns, show="headings", height=3)
current_table.heading("Waktu", text="Waktu")
current_table.heading("Suhu", text="Suhu (°C)")
current_table.heading("Kelembapan", text="Kelembapan (%)")
current_table.insert("", "end", iid="current", values=("---", "---", "---"))
current_table.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=15)

# Animasi
ani = FuncAnimation(fig, plot_real_time, interval=1000, cache_frame_data=False)
root.mainloop()
