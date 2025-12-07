"""
Configuration file containing C3PA Taxonomy definitions.
These descriptions serve as the ground truth for the LLM prompt.
"""

LABEL_DESCRIPTIONS = {
    "Updated Privacy Policy": (
        "Annotate any statement indicating when the policy was last updated or its effective date."
    ),
    "Categories of Personal Information Collected": (
        "Cal. Civ. Code § 1798.130(a)(5)(B)(i): A list of the categories of personal information collected in the preceding 12 months. "
        "ONLY the **types** of PI collected (e.g., Identifiers; Internet Activity), **not** how it’s used."
    ),
    "Categories of Personal Information Sold": (
        "Cal. Civ. Code § 1798.115(c)(1): The category or categories of personal information sold... or if the business has not sold PI, it shall disclose that fact."
    ),
    "Categories of Personal Information Shared / Disclosed": (
        "Cal. Civ. Code § 1798.115(c)(2): The category or categories of personal information disclosed for a business purpose."
    ),
    "Description of Right to Delete": (
        "Cal. Civ. Code § 1798.105(a): Annotate any statement granting or describing the right to delete personal information."
    ),
    "Description of Right to Correct Information": (
        "Cal. Civ. Code § 1798.106(a): Annotate any statement granting or describing the right to correct inaccurate information."
    ),
    "Description of Right to Know PI Collected": (
        "Cal. Civ. Code § 1798.110(a): Annotate statements describing the right to access or know what data has been collected."
    ),
    "Description of Right to Know PI sold / shared": (
        "Cal. Civ. Code § 1798.115(a): Annotate statements describing the right to know what data has been sold or shared."
    ),
    "Description of Right to Opt-out of sale of PI": (
        "Cal. Civ. Code § 1798.120(a): Annotate any instruction or statement granting the right to opt-out of sale or sharing."
    ),
    "Description of Right to Limit use of PI": (
        "CPRA § 1798.121(a): Annotate any statement describing the limitation on use of sensitive personal information."
    ),
    "Description of Right to Non-discrimination on exercising rights": (
        "Cal. Civ. Code § 1798.125(a)(1): Annotate assurance that exercising rights will not result in discrimination."
    ),
    "Methods to exercise rights": (
        "Cal. Civ. Code § 1798.130(a)(1): Annotate descriptions of how to submit requests (e.g., toll-free number, web form, email)."
    ),
}