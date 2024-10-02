namespace Yolo;

internal static class NearestNeighbourImageResizer
{
	public static void Resize<TPixel>(ReadOnlySpan2D<TPixel> source, Span2D<TPixel> target)
	{
		Vector2D<float> scaleFactor = (Vector2D<float>)source.Size / (Vector2D<float>)target.Size;
		for (int targetYPosition = 0; targetYPosition < target.Size.Y; targetYPosition++)
		{
			var sourceYPosition = (int)Math.Round(targetYPosition * scaleFactor.Y, MidpointRounding.ToZero);
			ReadOnlySpan<TPixel> sourceRow = source.Span.Slice(source.Size.X * sourceYPosition, source.Size.X);
			Span<TPixel> targetRow = target.Span.Slice(target.Size.X * targetYPosition, target.Size.X);
			for (int targetXPosition = 0; targetXPosition < target.Size.X; targetXPosition++)
			{
				var sourceXPosition = (int)Math.Round(targetXPosition * scaleFactor.X, MidpointRounding.ToZero);
				targetRow[targetXPosition] = sourceRow[sourceXPosition];
			}
		}
	}
}