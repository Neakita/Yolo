namespace TensorWeaver.OutputData;

public readonly struct Bounding
{
	public static Bounding Empty { get; } = new();

	public static Bounding operator *(Bounding bounding, Vector2D<int> vector) => new(
		bounding.Left * vector.X,
		bounding.Top * vector.Y,
		bounding.Right * vector.X,
		bounding.Bottom * vector.Y);

	public static Bounding operator /(Bounding bounding, Vector2D<int> vector) => new(
		bounding.Left / vector.X,
		bounding.Top / vector.Y,
		bounding.Right / vector.X,
		bounding.Bottom / vector.Y);

	public float Left { get; }
	public float Top { get; }
	public float Right { get; }
	public float Bottom { get; }
	public float Width => Right - Left;
	public float Height => Bottom - Top;
	public float Area => Width * Height;

	public Bounding(float left, float top, float right, float bottom)
	{
		Left = left;
		Top = top;
		Right = right;
		Bottom = bottom;
		if (Width < 0)
			throw new ArgumentOutOfRangeException($"{nameof(left)} and {nameof(right)} should be computed to {nameof(Width)} greater or equal to 0, but was {Width}", (Exception?)null);
		if (Width < 0)
			throw new ArgumentOutOfRangeException($"{nameof(top)} and {nameof(bottom)} should be computed to {nameof(Height)} greater or equal to 0, but was {Height}", (Exception?)null);
	}

	public override string ToString()
	{
		return $"{nameof(Left)}: {Left}, {nameof(Top)}: {Top}, {nameof(Right)}: {Right}, {nameof(Bottom)}: {Bottom}";
	}
}