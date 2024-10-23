using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;

namespace Yolo.Metadata;

internal class TensorInfo
{
	public TensorShape Input { get; }
	public string Output0Name { get; }
	public TensorShape Output0 { get; }
	public string? Output1Name { get; }
	public TensorShape? Output1 { get; }

	public TensorInfo(InferenceSession session)
	{
		var inputMetadata = session.InputMetadata;
		var outputMetadata = session.OutputMetadata;
		Guard.IsBetweenOrEqualTo(outputMetadata.Count, 1, 2);
		var inputName = session.InputNames[0];
		Output0Name = session.OutputNames[0];
		Input = new TensorShape(inputMetadata[inputName].Dimensions);
		Output0 = new TensorShape(outputMetadata[Output0Name].Dimensions);
		if (session.OutputNames.Count == 2)
		{
			Output1Name = session.OutputNames[1];
			Output1 = new TensorShape(outputMetadata[Output1Name].Dimensions);
		}
	}
}