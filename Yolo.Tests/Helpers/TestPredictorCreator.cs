using Microsoft.ML.OnnxRuntime;

namespace Yolo.Tests.Helpers;

internal static class TestPredictorCreator
{
	public static Predictor CreatePredictor(string modelFileName, bool useGpu)
	{
		var modelFilePath = Path.Combine("Models", modelFileName);
		var modelData = File.ReadAllBytes(modelFilePath);
		var sessionOptions = useGpu ? SessionOptions.MakeSessionOptionWithCudaProvider() : new SessionOptions();
		return new Predictor(modelData, sessionOptions);
	}
}