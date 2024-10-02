using CommunityToolkit.Diagnostics;

namespace Yolo;

public readonly ref struct Span2D<TPixel>
{
	public Vector2D<int> Size { get; }
	public Span<TPixel> Span { get; }

	public TPixel this[Vector2D<int> position]
	{
		get
		{
			Guard.IsInRange(position.X, 0, Size.X);
			Guard.IsInRange(position.Y, 0, Size.Y);
			return Span[position.X + position.Y * Size.X];
		}
		set
		{
			Guard.IsInRange(position.X, 0, Size.X);
			Guard.IsInRange(position.Y, 0, Size.Y);
			Span[position.X + position.Y * Size.X] = value;
		}
	}

	public Span2D(Vector2D<int> size, Span<TPixel> span)
	{
		Guard.IsEqualTo(size.X * size.Y, span.Length);
		Guard.IsGreaterThan(size.X, 0);
		Guard.IsGreaterThan(size.Y, 0);
		Size = size;
		Span = span;
	}
}