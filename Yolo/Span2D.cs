using CommunityToolkit.Diagnostics;

namespace Yolo;

public readonly ref struct Span2D<T>
{
	public Vector2D<int> Size => new(_width, Span.Length / _width);
	public Span<T> Span { get; }

	public T this[Vector2D<int> position]
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

	public Span2D(Vector2D<int> size, Span<T> span)
	{
		Guard.IsEqualTo(size.X * size.Y, span.Length);
		Guard.IsGreaterThan(size.X, 0);
		Guard.IsGreaterThan(size.Y, 0);
		_width = size.X;
		Span = span;
	}

	public unsafe Span2D(Vector2D<int> size, void* pointer)
	{
		Guard.IsGreaterThan(size.X, 0);
		Guard.IsGreaterThan(size.Y, 0);
		_width = size.X;
		Span = new Span<T>(pointer, size.X * size.Y);
	}

	private readonly int _width;
}