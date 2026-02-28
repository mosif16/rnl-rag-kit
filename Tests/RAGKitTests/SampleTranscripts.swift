import Foundation

enum SampleTranscripts {
    static let short = "The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration. This process is essential for all living organisms."

    static let medium = """
    Photosynthesis is the process by which green plants convert sunlight into chemical energy. \
    The process takes place in the chloroplasts, which contain chlorophyll. \
    Light energy is absorbed and used to convert carbon dioxide and water into glucose and oxygen. \
    This is one of the most important biological processes on Earth. \
    Without photosynthesis, most life forms would not exist. \
    The equation for photosynthesis is 6CO2 + 6H2O -> C6H12O6 + 6O2. \
    Plants use the glucose for energy and growth. \
    The oxygen produced is released into the atmosphere. \
    Animals and humans depend on this oxygen for respiration. \
    Photosynthesis occurs in two stages: the light reactions and the Calvin cycle.
    """

    static let long: String = {
        let sentences = [
            "The nervous system is a complex network that coordinates actions and sensory information.",
            "It consists of the central nervous system and the peripheral nervous system.",
            "Neurons are the basic functional units of the nervous system.",
            "They transmit information through electrical and chemical signals.",
            "The brain is the most complex organ in the human body.",
            "It contains approximately 86 billion neurons.",
            "Synapses are the junctions between neurons where signals are transmitted.",
            "Neurotransmitters are chemical messengers that cross the synaptic cleft.",
            "The spinal cord connects the brain to the rest of the body.",
            "Reflexes are automatic responses that do not require brain processing.",
            "The autonomic nervous system controls involuntary functions.",
            "The somatic nervous system controls voluntary movements.",
            "Sensory neurons carry information from receptors to the brain.",
            "Motor neurons carry commands from the brain to muscles.",
            "Interneurons connect sensory and motor neurons within the CNS.",
            "Myelin sheaths increase the speed of nerve impulse transmission.",
            "Multiple sclerosis is caused by damage to myelin sheaths.",
            "The blood-brain barrier protects the brain from harmful substances.",
            "Neuroplasticity allows the brain to reorganize itself throughout life.",
            "Sleep is essential for memory consolidation and brain health."
        ]
        return sentences.joined(separator: " ")
    }()

    static func repeated(sentences: Int) -> String {
        let base = "This is a test sentence about an important concept. "
        return String(repeating: base, count: sentences)
    }
}
