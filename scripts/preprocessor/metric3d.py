from ..supported_preprocessor import Preprocessor, PreprocessorParameter

class PreprocessorMetric3DDepth(Preprocessor):

    def __init__(self):
        super().__init__(name="depth_metric3d")
        self.tags = ["Depth"]
        self.slider_1 = PreprocessorParameter(
            minimum=0,
            maximum=360,
            step=0.1,
            value=60,
            label="Fov",
        )
        self.slider_2 = PreprocessorParameter(
            minimum=1,
            maximum=20,
            step=1,
            value=5,
            label="Iterations",
        )
        self.model = None
        
    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        if self.model is None:
            from annotator.metric3d import Metric3DDetector

            self.model = Metric3DDetector()

        result = self.model(
            input_image,
            new_fov=float(slider_1),
            iterations=int(slider_2),
            resulotion=resolution,
        )
        return result

class PreprocessorMetric3DNormal(Preprocessor):

    def __init__(self):
        super().__init__(name="normal_metric3d")
        self.tags = ["NormalMap"]
        self.slider_1 = PreprocessorParameter(
            minimum=0,
            maximum=360,
            step=0.1,
            value=60,
            label="Fov",
        )
        self.slider_2 = PreprocessorParameter(
            minimum=1,
            maximum=20,
            step=1,
            value=5,
            label="Iterations",
        )
        self.model = None
        
    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        if self.model is None:
            from annotator.normaldsine import NormalDsineDetector

            self.model = NormalDsineDetector()

        result = self.model(
            input_image,
            new_fov=float(slider_1),
            iterations=int(slider_2),
            resulotion=resolution,
        )
        return result

Preprocessor.add_supported_preprocessor(PreprocessorMetric3DDepth())
Preprocessor.add_supported_preprocessor(PreprocessorMetric3DNormal())