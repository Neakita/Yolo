namespace Yolo;

internal static class NearestNeighbourImageResizer
{
	public static void Resize<TPixel>(ReadOnlySpan2D<TPixel> source, Span2D<TPixel> target)
	{
		Vector2D<float> scaleFactor = (Vector2D<float>)source.Size / (Vector2D<float>)target.Size;
		for (int y = 0; y < target.Size.X; y++)
		for (int x = 0; x < target.Size.Y; x++)
		{
			Vector2D<int> targetPosition = new(x, y);
			Vector2D<int> sourcePosition = new(
				(int)Math.Round(targetPosition.X * scaleFactor.X, MidpointRounding.ToZero),
				(int)Math.Round(targetPosition.Y * scaleFactor.Y, MidpointRounding.ToZero));
			target[targetPosition] = source[sourcePosition];
		}
	}
}