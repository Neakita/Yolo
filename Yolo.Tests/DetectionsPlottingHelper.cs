using System.Collections.Immutable;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;

namespace Yolo.Tests;

internal static class DetectionsPlottingHelper
{
	public static Image Plot(Image image, Metadata metadata, IReadOnlyCollection<Detection> detections)
	{
		return image.Clone(processingContext => Plot(processingContext, metadata, detections));
	}

	private static void Plot(IImageProcessingContext processingContext, Metadata metadata, IReadOnlyCollection<Detection> detections)
	{
		foreach (var detection in detections)
			Plot(processingContext, metadata, detection);
	}

	private static void Plot(IImageProcessingContext processingContext, Metadata metadata, Detection detection)
	{
		var bounding = detection.Bounding;
		Plot(processingContext, bounding);
		PointF labelLocation = new(bounding.Left, bounding.Top);
		Plot(processingContext, metadata, detection.Classification, labelLocation);
	}

	private static void Plot(IImageProcessingContext processingContext, Bounding bounding)
	{
		RectangleF rectangle = new(bounding.Left, bounding.Top, bounding.Width, bounding.Height);
		processingContext.Draw(Color.Red, 2, rectangle);
	}

	private static void Plot(IImageProcessingContext processingContext, Metadata metadata, Classification classification, PointF labelLocation)
	{
		var label = $"{metadata.ClassesNames[classification.ClassId]}: {classification.Confidence:P1}";
		processingContext.DrawText(label, Font, Color.Red, labelLocation);
	}

	private static readonly Font Font;

	static DetectionsPlottingHelper()
	{
		foreach (var possibleFontFamilyName in PossibleFontFamilyNames)
		{
			if (!SystemFonts.TryGet(possibleFontFamilyName, out var fontFamily))
				continue;
			Font = fontFamily.CreateFont(12);
			return;
		}
		throw new InvalidOperationException("Can't find system font");
	}

	private static readonly ImmutableArray<string> PossibleFontFamilyNames = ["FreeMono"];
}