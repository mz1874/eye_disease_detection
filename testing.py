import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go

# ===================== 1. 加载 DICOM 序列 =====================
def load_dicom_series(root_folder):
    slices = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for f in filenames:
            path = os.path.join(dirpath, f)
            try:
                ds = pydicom.dcmread(path)
                slices.append(ds)
            except Exception:
                pass
    if len(slices) == 0:
        raise ValueError("No DICOM slices found!")

    # 打印第一个切片元信息
    first_slice = slices[0]
    print("=== First slice DICOM metadata ===")
    print(first_slice)

    # 形状筛选
    shapes = [s.pixel_array.shape for s in slices]
    most_common_shape = max(set(shapes), key=shapes.count)
    slices = [s for s in slices if s.pixel_array.shape == most_common_shape]

    # 排序
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    # 堆叠成 3D volume
    volume = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    # HU 校正
    for i, s in enumerate(slices):
        slope = float(getattr(s, 'RescaleSlope', 1))
        intercept = float(getattr(s, 'RescaleIntercept', 0))
        volume[i] = volume[i] * slope + intercept

    return volume, first_slice, slices

# ===================== 2. 获取 voxel spacing =====================
def get_spacing(slices):
    first = slices[0]
    dy, dx = map(float, first.PixelSpacing)
    z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
    z_positions = sorted(z_positions)
    dz = abs(z_positions[1] - z_positions[0])
    return (dz, dy, dx)

# ===================== 3. 窗宽窗位 =====================
def window_image(img, WL=-600, WW=1500):
    lower = WL - WW / 2
    upper = WL + WW / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / WW
    return img

# ===================== 4. 加载 XML 标注 =====================
def load_annotations(xml_path, volume_shape):
    mask_dict = {}
    if not os.path.exists(xml_path):
        return mask_dict

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for nodule in root.findall('readingSession'):
        doctor_id = nodule.attrib.get('user', 'doc')
        if doctor_id not in mask_dict:
            mask_dict[doctor_id] = np.zeros(volume_shape, dtype=np.uint8)

        for lesion in nodule.findall('unblindedReadNodule'):
            for roi in lesion.findall('roi'):
                z = int(float(roi.attrib['sliceIndex']))
                for point in roi.findall('point'):
                    x = int(float(point.attrib['x']))
                    y = int(float(point.attrib['y']))
                    if 0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]:
                        mask_dict[doctor_id][z, y, x] = 1
    return mask_dict

# ===================== 5. 2D Viewer =====================
class CTViewer2D:
    def __init__(self, volume, masks, meta):
        self.volume = volume
        self.masks = masks
        self.doctors = list(masks.keys())
        self.current_doctor_idx = 0
        self.idx = volume.shape[0] // 2
        self.meta = meta

        self.fig = plt.figure(figsize=(10,6))
        gs = self.fig.add_gridspec(1,2, width_ratios=[1,4], wspace=0.05)
        self.ax_z = self.fig.add_subplot(gs[0,0])
        self.ax_main = self.fig.add_subplot(gs[0,1])

        self.im = self.ax_main.imshow(window_image(self.volume[self.idx]), cmap='gray')
        self.mask_im = self.ax_main.imshow(self.get_current_mask(), cmap='Reds', alpha=0.4)
        self.ax_main.axis('off')

        self.z_thumb = np.max(self.volume, axis=1)
        self.ax_z.imshow(self.z_thumb, cmap='gray', origin='upper', aspect='auto')
        self.z_line = self.ax_z.axhline(self.idx, color='red', linewidth=1)
        self.ax_z.set_title("Z-axis")
        self.ax_z.axis('off')

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update_display()
        plt.show()

    def get_current_mask(self):
        if len(self.doctors) == 0:
            return np.zeros_like(self.volume[self.idx])
        return self.masks[self.get_current_doctor()][self.idx]

    def get_current_doctor(self):
        if len(self.doctors) == 0:
            return "None"
        return self.doctors[self.current_doctor_idx]

    def update_display(self):
        self.im.set_data(window_image(self.volume[self.idx]))
        self.mask_im.set_data(self.get_current_mask())
        self.ax_main.set_title(f"{self.meta.PatientID} Slice {self.idx} - Doctor: {self.get_current_doctor()}")
        self.z_line.set_ydata([self.idx, self.idx])
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'up':
            self.idx = min(self.idx + 1, self.volume.shape[0]-1)
        elif event.key == 'down':
            self.idx = max(self.idx - 1, 0)
        elif event.key == 'right':
            if len(self.doctors) > 0:
                self.current_doctor_idx = (self.current_doctor_idx + 1) % len(self.doctors)
        elif event.key == 'left':
            if len(self.doctors) > 0:
                self.current_doctor_idx = (self.current_doctor_idx - 1) % len(self.doctors)
        self.update_display()

    def on_scroll(self, event):
        if event.button == 'up':
            self.idx = min(self.idx + 1, self.volume.shape[0]-1)
        elif event.button == 'down':
            self.idx = max(self.idx - 1, 0)
        self.update_display()

    def on_click(self, event):
        if event.inaxes == self.ax_z:
            y = int(event.ydata)
            if 0 <= y < self.volume.shape[0]:
                self.idx = y
                self.update_display()

# ===================== 6. 3D Visualization =====================
def show_3d_surface(volume, threshold=-600):
    verts, faces, normals, values = measure.marching_cubes(volume, threshold)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, volume.shape[2])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[0])
    plt.title("3D Lung Surface")
    plt.show()

def show_3d_volume_plotly(volume):
    fig = go.Figure(data=go.Volume(
        x=np.arange(volume.shape[2]),
        y=np.arange(volume.shape[1]),
        z=np.arange(volume.shape[0]),
        value=volume.flatten(),
        opacity=0.1,
        surface_count=20,
    ))
    fig.update_layout(scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
    ))
    fig.show()

# ===================== 7. 主程序 =====================
if __name__ == "__main__":
    folder_path = "./LIDC-IDRI-0001/01-01-2000-NA-NA-30178"  # 修改为你的 DICOM 文件夹
    xml_path = os.path.join(folder_path, "069.xml")           # 标注 XML 路径

    volume, meta, slices = load_dicom_series(folder_path)
    print("Volume shape:", volume.shape)
    print("HU range:", volume.min(), volume.max())
    print("Patient ID:", meta.PatientID)

    masks = load_annotations(xml_path, volume.shape)
    if masks:
        print("Loaded doctors:", list(masks.keys()))
    else:
        print("No annotation XML found!")

    spacing = get_spacing(slices)
    print("Voxel spacing (dz, dy, dx) in mm:", spacing)

    # ===== 2D 浏览器 =====
    CTViewer2D(volume, masks, meta)

    # ===== 3D 可视化 =====
    # 1) marching cubes 表面模型
    show_3d_surface(volume, threshold=-600)

    # 2) Plotly volume rendering
    show_3d_volume_plotly(volume)
