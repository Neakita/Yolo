using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;
using TensorWeaver.OutputData;

namespace TensorWeaver.Tests.Inference.Detectors;

internal sealed class DetectionsPlotter : ResultHandler<IEnumerable<Detection>>
{
	public DetectionsPlotter(ImageInfo imageInfo, IReadOnlyList<string> classesNames, string plottedFileName)
	{
		_imageInfo = imageInfo;
		_classesNames = classesNames;
		_plottedFileName = plottedFileName;
	}

	public async Task HandleResultAsync(IEnumerable<Detection> detections, CancellationToken cancellationToken)
	{
		using var image = await _imageInfo.LoadAsync(cancellationToken);
		var plottedImage = Plot(image, detections);
		var plottedFilePath = Path.Combine("plotted-images", _plottedFileName);
		Directory.CreateDirectory("plotted-images");
		await plottedImage.SaveAsync(plottedFilePath, cancellationToken: cancellationToken);
	}

	private Image Plot(Image image, IEnumerable<Detection> detections)
	{
		return image.Clone(processingContext => Plot(processingContext, detections));
	}

	private void Plot(
		IImageProcessingContext processingContext,
		IEnumerable<Detection> detections)
	{
		foreach (var detection in detections)
			Plot(processingContext, detection);
	}

	private readonly ImageInfo _imageInfo;
	private readonly IReadOnlyList<string> _classesNames;
	private readonly string _plottedFileName;

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