import zivid


class Zivid:
    def __init__(self, width = 1224, height = 1024):
        # --- Store the core camera properties ---
        self.width = width
        self.height = height
        # --- You can initialize other things here ---
        self.camera_handle = None # Placeholder for the actual Zivid camera connection
        self.is_connected = False
        self.app = None

        print(f"ZividCamera instance created for a {width}x{height} camera.")
    def subsampledSettingsForCamera(camera: zivid.Camera) -> zivid.Settings:
        settings_subsampled = zivid.Settings(
        acquisitions=[zivid.Settings.Acquisition()],
        color=zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()]),
        )
        model = camera.info.model
        if (
        model is zivid.CameraInfo.Model.zividTwo
        or model is zivid.CameraInfo.Model.zividTwoL100
        or model is zivid.CameraInfo.Model.zivid2PlusM130
        or model is zivid.CameraInfo.Model.zivid2PlusM60
        or model is zivid.CameraInfo.Model.zivid2PlusL110
        ):
            settings_subsampled.sampling.pixel = zivid.Settings.Sampling.Pixel.blueSubsample2x2
            settings_subsampled.color.sampling.pixel = zivid.Settings2D.Sampling.Pixel.blueSubsample2x2
        elif (
        model is zivid.CameraInfo.Model.zivid2PlusMR130
        or model is zivid.CameraInfo.Model.zivid2PlusMR60
        or model is zivid.CameraInfo.Model.zivid2PlusLR110
        ):
            settings_subsampled.sampling.pixel = zivid.Settings.Sampling.Pixel.by2x2
            settings_subsampled.color.sampling.pixel = zivid.Settings2D.Sampling.Pixel.by2x2
        else:
            raise ValueError(f"Unhandled enum value {model}")

        return settings_subsampled        
    def connect(self):
        self.app = zivid.Application()
        print("Connecting to camera")
        camera = self.app.connect_camera()

        print("Getting camera intrinsics")
        intrinsics = zivid.experimental.calibration.intrinsics(camera)
        print(intrinsics)

        output_file = "Intrinsics.yml"
        print(f"Saving camera intrinsics to file: {output_file}")
        intrinsics.save(output_file)