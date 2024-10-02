using System.Numerics;

namespace Yolo;

public readonly struct Vector2D<T> : IEquatable<Vector2D<T>> where T : INumber<T>, IConvertible
{
	public static explicit operator Vector2D<float>(Vector2D<T> value)
	{
		return new Vector2D<float>(value.X.ToSingle(null), value.Y.ToSingle(null));
	}

	public static explicit operator Vector2D<int>(Vector2D<T> value)
	{
		return new Vector2D<int>(value.X.ToInt32(null), value.Y.ToInt32(null));
	}

	public static bool operator ==(Vector2D<T> left, Vector2D<T> right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Vector2D<T> left, Vector2D<T> right)
	{
		return !left.Equals(right);
	}

	public static Vector2D<T> operator +(Vector2D<T> first, Vector2D<T> second)
	{
		return new Vector2D<T>(first.X + second.X, first.Y + second.Y);
	}

	public static Vector2D<T> operator -(Vector2D<T> first, Vector2D<T> second)
	{
		return new Vector2D<T>(first.X - second.X, first.Y - second.Y);
	}

	public static Vector2D<T> operator +(Vector2D<T> first, T second)
	{
		return new Vector2D<T>(first.X + second, first.Y + second);
	}

	public static Vector2D<T> operator -(Vector2D<T> first, T second)
	{
		return new Vector2D<T>(first.X - second, first.Y - second);
	}

	public static Vector2D<T> operator *(Vector2D<T> first, T second)
	{
		return new Vector2D<T>(first.X * second, first.Y * second);
	}

	public static Vector2D<T> operator /(Vector2D<T> first, T second)
	{
		return new Vector2D<T>(first.X / second, first.Y / second);
	}

	public static Vector2D<T> operator *(Vector2D<T> first, Vector2D<T> second)
	{
		return new Vector2D<T>(first.X * second.X, first.Y * second.Y);
	}

	public static Vector2D<T> operator /(Vector2D<T> first, Vector2D<T> second)
	{
		return new Vector2D<T>(first.X / second.X, first.Y / second.Y);
	}

	public static Vector2D<T> Zero { get; } = new(T.Zero, T.Zero);

	public T X { get; }
	public T Y { get; }

	public Vector2D(T x, T y)
	{
		X = x;
		Y = y;
	}

	public Vector2D<T> WithX(T value)
	{
		return new Vector2D<T>(value, Y);
	}

	public Vector2D<T> WithY(T value)
	{
		return new Vector2D<T>(X, value);
	}

	public Vector2D<T> Clamp(Vector2D<T> min, Vector2D<T> max)
	{
		return new Vector2D<T>(T.Clamp(X, min.X, max.X), T.Clamp(Y, min.Y, max.Y));
	}

	public bool Equals(Vector2D<T> other)
	{
		return EqualityComparer<T>.Default.Equals(X, other.X) && EqualityComparer<T>.Default.Equals(Y, other.Y);
	}

	public override bool Equals(object? obj)
	{
		return obj is Vector2D<T> other && Equals(other);
	}

	public override int GetHashCode()
	{
		return HashCode.Combine(X, Y);
	}
}