using System.Collections.Immutable;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TensorWeaver;
using TensorWeaver.Metadata;
using TensorWeaver.OutputData;

namespace TensorWeaver.Tests.Helpers;

internal static class DetectionsPlottingHelper
{
	public static Image Plot(Image image, ModelMetadata metadata, IReadOnlyCollection<Detection> detections)
	{
		return image.Clone(processingContext => Plot(processingContext, metadata, detections));
	}

	public static void Plot(
		IImageProcessingContext processingContext,
		ModelMetadata metadata,
		IEnumerable<Detection> detections)
	{
		foreach (var detection in detections)
			Plot(processingContext, metadata, detection);
	}

	private static void Plot(IImageProcessingContext processingContext, ModelMetadata metadata, Detection detection)
	{
		var bounding = detection.Bounding;
		var imageSize = processingContext.GetCurrentSize();
		bounding *= new Vector2D<int>(imageSize.Width, imageSize.Height);
		Plot(processingContext, bounding);
		PointF labelLocation = new(bounding.Left, bounding.Top);
		Plot(processingContext, metadata, detection.Classification, labelLocation);
	}

	private static void Plot(IImageProcessingContext processingContext, Bounding bounding)
	{
		RectangleF rectangle = new(bounding.Left, bounding.Top, bounding.Width, bounding.Height);
		processingContext.Draw(Color.Red, 1, rectangle);
	}

	private static void Plot(IImageProcessingContext processingContext, ModelMetadata metadata, Classification classification, PointF labelLocation)
	{
		var label = $"{metadata.ClassesNames[classification.ClassId]}: {classification.Confidence:P1}";
		processingContext.DrawText(label, Font, new Color(new Rgb24(0, 255, 0)), labelLocation);
	}

	private static readonly Font Font;

	static DetectionsPlottingHelper()
	{
		foreach (var possibleFontFamilyName in PossibleFontFamilyNames)
		{
			if (!SystemFonts.TryGet(possibleFontFamilyName, out var fontFamily))
				continue;
			Font = fontFamily.CreateFont(12, FontStyle.Bold);
			return;
		}
		throw new InvalidOperationException("Can't find system font");
	}

	private static readonly ImmutableArray<string> PossibleFontFamilyNames = ["FreeMono", "Noto Sans", "Arial"];
}