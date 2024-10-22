using CommunityToolkit.HighPerformance;

namespace Yolo;

public static class NearestNeighbourImageResizer
{
	public static void Resize<TPixel>(ReadOnlySpan2D<TPixel> source, Span2D<TPixel> target)
	{
		Vector2D<float> scaleFactor = new((float)source.Width / target.Width, (float)source.Height / target.Height);
		Span<int> targetToSourcePositionLookupTable = stackalloc int[target.Width];
		for (int targetXPosition = 0; targetXPosition < target.Width; targetXPosition++)
		{
			var sourceXPosition = (int)Math.Round(targetXPosition * scaleFactor.X, MidpointRounding.ToZero);
			targetToSourcePositionLookupTable[targetXPosition] = sourceXPosition;
		}
		for (int targetYPosition = 0; targetYPosition < target.Height; targetYPosition++)
		{
			var sourceYPosition = (int)Math.Round(targetYPosition * scaleFactor.Y, MidpointRounding.ToZero);
			ReadOnlySpan<TPixel> sourceRow = source.GetRowSpan(sourceYPosition);
			Span<TPixel> targetRow = target.GetRowSpan(targetYPosition);
			for (int targetXPosition = 0; targetXPosition < target.Width; targetXPosition++)
			{
				var sourceXPosition = targetToSourcePositionLookupTable[targetXPosition];
				targetRow[targetXPosition] = sourceRow[sourceXPosition];
			}
		}
	}
}