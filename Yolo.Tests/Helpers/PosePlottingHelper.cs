using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;

namespace Yolo.Tests.Helpers;

public static class PosePlottingHelper
{
	public static Image Plot(Image image, Metadata metadata, IReadOnlyCollection<Pose> poses)
	{
		return image.Clone(processingContext => Plot(processingContext, metadata, poses));
	}

	private static void Plot(
		IImageProcessingContext processingContext,
		Metadata metadata,
		IReadOnlyCollection<Pose> poses)
	{
		DetectionsPlottingHelper.Plot(processingContext, metadata, poses.Select(pose => pose.Detection));
		foreach (var pose in poses)
			Plot(processingContext, metadata, pose);
	}

	private static void Plot(IImageProcessingContext processingContext, Metadata metadata, Pose pose)
	{
		foreach (var keyPoint in pose.KeyPoints)
			Plot(processingContext, metadata, keyPoint);
	}

	private static void Plot(IImageProcessingContext processingContext, Metadata metadata, KeyPoint keyPoint)
	{
		Vector2D<float> position = keyPoint.Position;
		position *= metadata.ImageSize.ToSingle();
		EllipsePolygon ellipse = new(position.X, position.Y, 1);
		processingContext.Draw(Color.Blue, 3, ellipse);
	}
}