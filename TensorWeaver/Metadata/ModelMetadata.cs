using System.Collections.Immutable;
using Microsoft.ML.OnnxRuntime;

namespace TensorWeaver.Metadata;

public class ModelMetadata
{
	public ImmutableArray<string> ClassesNames { get; }
	public Vector2D<int> ImageSize
	{
		get
		{
			var dimensions = _session.InputMetadata.Values.Single().Dimensions;
			return new Vector2D<int>(dimensions[3], dimensions[2]);
		}
	}
	public byte Version { get; }
	public Task Task { get; }

	internal ModelMetadata(InferenceSession session)
	{
		_session = session;
		var metadata = session.ModelMetadata.CustomMetadataMap;
		Task = metadata["task"] switch
		{
			"classify" => Task.Classify,
			"detect" => Task.Detect,
			"obb" => Task.Obb,
			"pose" => Task.Pose,
			"segment" => Task.Segment,
			_ => throw new ArgumentOutOfRangeException()
		};
		ClassesNames = ParseNames(metadata["names"]);
		Version = DetectVersion(session);
	}

	private readonly InferenceSession _session;

	private byte DetectVersion(InferenceSession session)
	{
		// YOLOv10 output shape => [<batch>, 300, 6]
		if (Task == Task.Detect && session.OutputMetadata.Values.First().Dimensions[2] == 6)
			return 10;
		return 8;
	}

	private static ImmutableArray<string> ParseNames(string str)
	{
		str = str[1..^1];
		var split = str.Split(", ");
		var count = split.Length;
		// ReSharper disable once CollectionNeverUpdated.Local
		var builder = ImmutableArray.CreateBuilder<string>(count);
		builder.Count = count;
		foreach (var value in split)
		{
			var valueSplit = value.Split(": ");
			var id = int.Parse(valueSplit[0]);
			var name = valueSplit[1][1..^1].Replace('_', ' ');
			builder[id] = name;
		}
		return builder.DrainToImmutable();
	}
}