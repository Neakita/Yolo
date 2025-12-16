#pragma warning disable CS8618 // Nullability

using BenchmarkDotNet.Attributes;
using CommunityToolkit.Diagnostics;
using Compunet.YoloSharp;
using Compunet.YoloSharp.Data;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace TensorWeaver.Benchmark;

public partial class YoloClassificationBenchmark
{
	[GlobalSetup(Target = nameof(CompunetInference))]
	public void SetupCompunet()
	{
		var options = new YoloPredictorOptions
		{
			UseCuda = false,
			SessionOptions = ExecutionProvider.CreateSessionOptions()
		};
		_compunetPredictor = new YoloPredictor(Model.FilePath, options);
		_imageForCompunet = ImageInfo.Load<Rgb24>();
	}

	[GlobalCleanup(Target = nameof(CompunetInference))]
	public void CleanUpCompunet()
	{
		_compunetPredictor.Dispose();
		_imageForCompunet.Dispose();
	}

	[Benchmark]
	public YoloResult<Classification> CompunetInference()
	{
		var result = _compunetPredictor.Classify(_imageForCompunet);
		Guard.IsGreaterThan(result.Count, 0);
		return result;
	}

	private YoloPredictor _compunetPredictor;
	private Image<Rgb24> _imageForCompunet;
}