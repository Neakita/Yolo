using FluentAssertions;
using Xunit.Abstractions;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;

namespace Yolo.Tests;

public class DetectionTests
{
	// 2 persons are clearly visible and 2 more are only partially.
	// bus is huge object on the image and locates in the center
	// the sign is barely noticeable
	private const string BusExpectedDetections = "person:2-4,bus:1,stop sign:0-1";
	private static readonly Rgb24InputProcessor Rgb24InputProcessor = new();

	public DetectionTests(ITestOutputHelper testOutputHelper)
	{
		_testOutputHelper = testOutputHelper;
	}

	private readonly ITestOutputHelper _testOutputHelper;

	[Theory]
	[InlineData("yolov8n-uint8.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov8n-uint8.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov8n160fp32.onnx", "bus160.png", BusExpectedDetections, false)]
	[InlineData("yolov8n160fp32.onnx", "bus160.png", BusExpectedDetections, true)]
	[InlineData("yolov8n224fp32.onnx", "bus224.png", BusExpectedDetections, false)]
	[InlineData("yolov8n224fp32.onnx", "bus224.png", BusExpectedDetections, true)]
	[InlineData("yolov8n320fp32.onnx", "bus320.png", BusExpectedDetections, false)]
	[InlineData("yolov8n320fp32.onnx", "bus320.png", BusExpectedDetections, true)]
	[InlineData("yolov8n480fp32.onnx", "bus480.png", BusExpectedDetections, false)]
	[InlineData("yolov8n480fp32.onnx", "bus480.png", BusExpectedDetections, true)]
	[InlineData("yolov8n640fp32.onnx", "bus640.png", BusExpectedDetections, false)]
	[InlineData("yolov8n640fp32.onnx", "bus640.png", BusExpectedDetections, true)]
	[InlineData("yolov8n800fp32.onnx", "bus800.png", BusExpectedDetections, false)]
	[InlineData("yolov8n800fp32.onnx", "bus800.png", BusExpectedDetections, true)]
	public void ShouldDetectFromImageOfSameSize(
		string modelFileName,
		string imageFileName,
		string expectedDetections,
		bool useGpu)
	{
		Predictor predictor = TestPredictorCreator.CreatePredictor(modelFileName, useGpu);
		V8DetectionProcessor outputProcessor = new(predictor.Metadata);
		var imageData = TestImageLoader.LoadImageData(imageFileName);
		var detections = predictor.Predict(imageData.Span2D, Rgb24InputProcessor, outputProcessor);
		AssertPrediction(predictor.Metadata, expectedDetections, detections);
		WriteDetections(_testOutputHelper, predictor.Metadata, detections);
	}

	private static void AssertPrediction(
		Metadata metadata,
		string expectedDetections,
		IReadOnlyList<Detection> actualDetections)
	{
		var expectations = ParseExpectations(expectedDetections).ToList();
		var countedDetections = CountDetections(metadata, actualDetections)
			.ToList();
		foreach (var expectation in expectations)
		{
			var detectionsCount = countedDetections.FirstOrDefault(detection => detection.className == expectation.ClassName).count;
			detectionsCount.Should().BeInRange(expectation.MinimumCount, expectation.MaximumCount);
			if (detectionsCount > 0)
				countedDetections.Remove((expectation.ClassName, detectionsCount));
		}
		countedDetections.Should().BeEmpty();
	}

	private static IEnumerable<PredictionExpectation> ParseExpectations(string expectedPrediction)
	{
		return expectedPrediction.Split(',').Select(ParseExpectation);
	}

	private static PredictionExpectation ParseExpectation(string expectation)
	{
		var split = expectation.Split(':');
		var className = split[0];
		var count = split[1];
		var countSplit = count.Split('-');
		if (countSplit.Length == 1)
		{
			var parsedCount = int.Parse(countSplit[0]);
			return new PredictionExpectation(className, parsedCount, parsedCount);
		}

		var parsedMinimum = int.Parse(countSplit[0]);
		var parsedMaximum = int.Parse(countSplit[1]);
		return new PredictionExpectation(className, parsedMinimum, parsedMaximum);
	}

	private static void WriteDetections(ITestOutputHelper outputHelper, Metadata metadata, IReadOnlyCollection<Detection> detections)
	{
		foreach (var detection in detections)
			outputHelper.WriteLine($"{metadata.ClassesNames[detection.ClassId]}: {detection.Confidence:P1}");
	}

	private static IEnumerable<(string className, int count)> CountDetections(Metadata metadata, IEnumerable<Detection> actualDetections)
	{
		return actualDetections
			.GroupBy(detection => detection.ClassId)
			.Select(group => (className: metadata.ClassesNames[group.Key], count: group.Count()));
	}
}