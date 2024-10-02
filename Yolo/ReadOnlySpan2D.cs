using CommunityToolkit.Diagnostics;

namespace Yolo;

public readonly ref struct ReadOnlySpan2D<TPixel>
{
	public static implicit operator ReadOnlySpan2D<TPixel>(Span2D<TPixel> span2D)
	{
		return new ReadOnlySpan2D<TPixel>(span2D.Size, span2D.Span);
	}

	public Vector2D<int> Size { get; }
	public ReadOnlySpan<TPixel> Span { get; }

	public TPixel this[Vector2D<int> position]
	{
		get
		{
			Guard.IsInRange(position.X, 0, Size.X);
			Guard.IsInRange(position.Y, 0, Size.Y);
			return Span[position.X + position.Y * Size.X];
		}
	}

	public ReadOnlySpan2D(Vector2D<int> size, Span<TPixel> span)
	{
		Guard.IsEqualTo(size.X * size.Y, span.Length);
		Guard.IsGreaterThan(size.X, 0);
		Guard.IsGreaterThan(size.Y, 0);
		Size = size;
		Span = span;
	}
}