using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;
using TensorWeaver.OutputData;

namespace TensorWeaver.Tests.Inference.Classificators;

public sealed class ClassificationsPlotter : ResultHandler<IEnumerable<Classification>>
{
	public ClassificationsPlotter(ImageInfo imageInfo, IReadOnlyList<string> classesNames, string plottedFileName)
	{
		_imageInfo = imageInfo;
		_classesNames = classesNames;
		_plottedFileName = plottedFileName;
	}

	public async Task HandleResultAsync(IEnumerable<Classification> classifications, CancellationToken cancellationToken)
	{
		using var image = await _imageInfo.LoadAsync(cancellationToken);
		var plottedImage = Plot(image, classifications);
		var plottedFilePath = Path.Combine("plotted-images", _plottedFileName);
		Directory.CreateDirectory("plotted-images");
		await plottedImage.SaveAsync(plottedFilePath, cancellationToken: cancellationToken);
	}

	private readonly ImageInfo _imageInfo;
	private readonly IReadOnlyList<string> _classesNames;
	private readonly string _plottedFileName;

	private Image Plot(Image image, IEnumerable<Classification> classifications)
	{
		return image.Clone(processingContext => Plot(processingContext, classifications));
	}

	private void Plot(
		IImageProcessingContext processingContext,
		IEnumerable<Classification> classifications)
	{
		float labelLocation = 0;
		foreach (var classification in classifications)
		{
			Plot(processingContext, classification, new PointF(0, labelLocation));
			labelLocation += 10;
		}
	}

	private void Plot(IImageProcessingContext processingContext, Classification classification, PointF labelLocation)
	{
		var label = $"{_classesNames[classification.ClassId]}: {classification.Confidence:P1}";
		processingContext.DrawText(label, FontProvider.Font, Color.Black, labelLocation);
	}
}