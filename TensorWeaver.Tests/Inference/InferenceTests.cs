using SixLabors.ImageSharp.PixelFormats;
using TensorWeaver.ImageSharp;
using TensorWeaver.InputProcessing;
using Xunit;

namespace TensorWeaver.Tests.Inference;

public sealed class InferenceTests
{
	public static IEnumerable<TestCase> Cases => TestCases.All;

	[Theory, CombinatorialData]
	public async Task ShouldInferenceAsync([CombinatorialMemberData(nameof(Cases))] TestCase testCase)
	{
		using var sessionOptions = testCase.ExecutionProvider.Factory();
		var modelData = await testCase.Model.ReadDataAsync(TestContext.Current.CancellationToken);
		using var predictor = new Predictor(modelData, sessionOptions);
		var pixels = await testCase.Image.GetPixelsAsync<Argb32>();
		predictor.SetInput(pixels.Span, InputProcessor);
		predictor.Predict();
		await testCase.OutputHandler.HandleOutputAsync(predictor, TestContext.Current.CancellationToken);
	}

	private static readonly ResizingInputProcessor<Argb32> InputProcessor = ImageSharpInputProcessors.Argb32.WithResizing();
}