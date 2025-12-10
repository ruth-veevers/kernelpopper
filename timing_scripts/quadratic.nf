
project_dir = projectDir

process one_quadratic {
	publishDir "${project_dir}", mode: 'copy', overwrite: true
	label 'usearray'
	input:
	tuple val(n_features), val(seed)
	
	output:
	path "result_quadratic_${n_features}_${seed}.txt"
	
	script:
	"""
	python ${project_dir}/scripts.py q $n_features $seed
	"""
}

workflow {
	main:
		feature_channel=Channel.of(101..149)
		seed_channel=Channel.of(0..9)
		one_quadratic(feature_channel.combine(seed_channel))
}