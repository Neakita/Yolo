using CommunityToolkit.Diagnostics;
using FluentAssertions;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Yolo.ImageSharp;
using Yolo.OutputProcessing;
using Path = System.IO.Path;

namespace Yolo.Tests;

public sealed class PoseTests
{
	[Theory]
	[InlineData("bus640.png", "person:4")]
	public void ShouldEstimate(string imageFileName, string expectedResults)
	{
		Predictor predictor = new(File.ReadAllBytes("Models/yolov8n-pose-uint8.onnx"), new SessionOptions());
		var imageFilePath = Path.Combine("Images", imageFileName);
		var image = Image.Load<Rgb24>(imageFilePath);
		Guard.IsTrue(image.DangerousTryGetSinglePixelMemory(out var data));
		var result = predictor.Predict(new ReadOnlySpan2D<Rgb24>(new Vector2D<int>(image.Width, image.Height), data.Span), new Rgb24InputProcessor(), new V8PoseProcessor(predictor));
		Draw(image, result);
		Directory.CreateDirectory(Path.Combine("Images", "Plotted"));
		image.Save(Path.Combine("Images", "Plotted", imageFileName));
		var stringResults = result.GroupBy(x => x.Detection.Classification.ClassId)
			.OrderByDescending(x => x.Count())
			.Select(x => $"{predictor.Metadata.ClassesNames[x.Key]}:{x.Count()}");
		var stringResult = string.Join(',', stringResults);
		stringResult.Should().Be(expectedResults);
	}

	private static void Draw(Image image, IEnumerable<Pose> poses)
	{
		foreach (var pose in poses)
			Draw(image, pose);
	}

	private static void Draw(Image image, Pose pose)
	{
		Draw(image, pose.Detection.Bounding);
		foreach (var keyPoint in pose.KeyPoints)
			Draw(image, keyPoint);
	}

	private static void Draw(Image image, Bounding bounding)
	{
		RectangleF rectangle = new(bounding.Left, bounding.Top, bounding.Width, bounding.Height);
		image.Mutate(x => x.Draw(Color.Red, 1, rectangle));
	}

	private static void Draw(Image image, KeyPoint keyPoint)
	{
		EllipsePolygon ellipse = new(keyPoint.Position.X, keyPoint.Position.Y, 1);
		image.Mutate(x => x.Draw(Color.Blue, 3, ellipse));
	}
}