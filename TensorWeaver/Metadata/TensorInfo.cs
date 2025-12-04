using Microsoft.ML.OnnxRuntime;

namespace TensorWeaver.Metadata;

internal class TensorInfo
{
	public static TensorInfo GetInputInfo(InferenceSession session)
	{
		var (inputName, inputMetadata) = session.InputMetadata.Single();
		return new TensorInfo
		{
			Name = inputName,
			Shape = new TensorShape(inputMetadata.Dimensions)
		};
	}

	public static IReadOnlyList<TensorInfo> GetOutputInfo(InferenceSession session)
	{
		var result = new List<TensorInfo>(session.OutputNames.Count);
		var outputMetadata = session.OutputMetadata;
		foreach (var outputName in session.OutputNames)
		{
			var info = new TensorInfo
			{
				Name = outputName,
				Shape = new TensorShape(outputMetadata[outputName].Dimensions)
			};
			result.Add(info);
		}
		return result;
	}

	public required string Name { get; init; }
	public TensorShape Shape { get; init; }
}