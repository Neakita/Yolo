using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;
using TensorWeaver.OutputData;
using Path = System.IO.Path;

namespace TensorWeaver.Tests.Inference.Posers;

internal sealed class PosesPlotter : ResultHandler<IEnumerable<Pose>>
{
	public PosesPlotter(ImageInfo imageInfo, IReadOnlyList<string> classesNames, string plottedFileName)
	{
		_imageInfo = imageInfo;
		_classesNames = classesNames;
		_plottedFileName = plottedFileName;
	}

	public async Task HandleResultAsync(IEnumerable<Pose> poses, CancellationToken cancellationToken)
	{
		using var image = await _imageInfo.LoadAsync(cancellationToken);
		var plottedImage = Plot(image, poses);
		var plottedFilePath = Path.Combine("plotted-images", _plottedFileName);
		Directory.CreateDirectory("plotted-images");
		await plottedImage.SaveAsync(plottedFilePath, cancellationToken: cancellationToken);
	}

	private readonly ImageInfo _imageInfo;
	private readonly IReadOnlyList<string> _classesNames;
	private readonly string _plottedFileName;

	private Image Plot(Image image, IEnumerable<Pose> poses)
	{
		return image.Clone(processingContext => Plot(processingContext, poses));
	}

	private void Plot(
		IImageProcessingContext processingContext,
		IEnumerable<Pose> poses)
	{
		foreach (var pose in poses)
			Plot(processingContext, pose);
	}

	private void Plot(IImageProcessingContext processingContext, Pose pose)
	{
		Plot(processingContext, pose.Detection);
		foreach (var keyPoint in pose.KeyPoints)
			Plot(processingContext, keyPoint);
	}

	private void Plot(IImageProcessingContext processingContext, KeyPoint keyPoint)
	{
		Vector2D<float> position = keyPoint.Position;
		var currentSize = processingContext.GetCurrentSize();
		position *= new Vector2D<float>(currentSize.Width, currentSize.Height);
		EllipsePolygon ellipse = new(position.X, position.Y, 1);
		processingContext.Draw(Color.Blue, 3, ellipse);
	}

	private void Plot(IImageProcessingContext processingContext, Detection detection)
	{
		var bounding = detection.Bounding;
		var imageSize = processingContext.GetCurrentSize();
		bounding *= new Vector2D<int>(imageSize.Width, imageSize.Height);
		Plot(processingContext, bounding);
		PointF labelLocation = new(bounding.Left, bounding.Top);
		Plot(processingContext, detection.Classification, labelLocation);
	}

	private static void Plot(IImageProcessingContext processingContext, Bounding bounding)
	{
		RectangleF rectangle = new(bounding.Left, bounding.Top, bounding.Width, bounding.Height);
		processingContext.Draw(Color.Red, 1, rectangle);
	}

	private void Plot(IImageProcessingContext processingContext, Classification classification, PointF labelLocation)
	{
		var label = $"{_classesNames[classification.ClassId]}: {classification.Confidence:P1}";
		processingContext.DrawText(label, FontProvider.Font, Color.Red, labelLocation);
	}
}