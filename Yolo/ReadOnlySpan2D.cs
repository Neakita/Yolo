using CommunityToolkit.Diagnostics;

namespace Yolo;

public readonly ref struct ReadOnlySpan2D<T>
{
	public static implicit operator ReadOnlySpan2D<T>(Span2D<T> span2D)
	{
		return new ReadOnlySpan2D<T>(span2D.Size, span2D.Span);
	}

	public Vector2D<int> Size => new(_width, Span.Length / _width);
	public ReadOnlySpan<T> Span { get; }

	public T this[Vector2D<int> position]
	{
		get
		{
			Guard.IsInRange(position.X, 0, Size.X);
			Guard.IsInRange(position.Y, 0, Size.Y);
			return Span[position.X + position.Y * Size.X];
		}
	}

	public ReadOnlySpan2D(Vector2D<int> size, ReadOnlySpan<T> span)
	{
		Guard.IsEqualTo(size.X * size.Y, span.Length);
		Guard.IsGreaterThan(size.X, 0);
		Guard.IsGreaterThan(size.Y, 0);
		_width = size.X;
		Span = span;
	}

	private readonly int _width;
}