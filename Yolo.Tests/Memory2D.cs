using CommunityToolkit.Diagnostics;

namespace Yolo.Tests;

public readonly struct Memory2D<T>
{
	public Vector2D<int> Size => new(_width, Memory.Length / _width);
	public Span2D<T> Span2D => new(Size, Memory.Span);
	public Memory<T> Memory { get; }

	public Memory2D(Vector2D<int> size, Memory<T> memory)
	{
		Guard.IsEqualTo(memory.Length, size.X * size.Y);
		Guard.IsGreaterThan(size.X, 0);
		Guard.IsGreaterThan(size.Y, 0);
		_width = size.X;
		Memory = memory;
	}

	private readonly int _width;
}