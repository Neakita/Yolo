using CommunityToolkit.Diagnostics;

namespace Yolo.Tests;

public readonly struct ReadOnlyMemory2D<T>
{
	public static implicit operator ReadOnlyMemory2D<T>(Memory2D<T> memory2D)
	{
		return new ReadOnlyMemory2D<T>(memory2D.Size, memory2D.Memory);
	}

	public Vector2D<int> Size => new(_width, Memory.Length / _width);
	public ReadOnlySpan2D<T> Span2D => new(Size, Memory.Span);
	public ReadOnlyMemory<T> Memory { get; }

	public ReadOnlyMemory2D(Vector2D<int> size, Memory<T> memory)
	{
		Guard.IsEqualTo(memory.Length, size.X * size.Y);
		Guard.IsGreaterThan(size.X, 0);
		Guard.IsGreaterThan(size.Y, 0);
		_width = size.X;
		Memory = memory;
	}

	private readonly int _width;
}