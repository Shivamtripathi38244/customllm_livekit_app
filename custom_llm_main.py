from datetime import datetime
import decimal
import os
import logging
import json
import re
import uuid
from difflib import get_close_matches

# Import LiveKit modules
from aiohttp import ClientError
from fastapi import requests
from livekit import agents
from livekit.agents.voice import ModelSettings
from livekit.agents import AgentSession, Agent, RoomInputOptions, Agent, llm, FunctionTool
from typing import Any, AsyncIterable, Dict, List
from livekit.plugins import openai, silero
from livekit.plugins import azure
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import boto3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(".env")

# Validate required environment variables
required_env_vars = [
    "AZURE_API_KEY",
    "AZURE_SPEECH_KEY",
    "LIVEKIT_URL"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Get API keys from environment variables
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-10-21")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "australiaeast")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
AZURE_ENDPOINT=os.getenv("AZURE_ENDPOINT")

logger.info(f"Using LiveKit URL: {LIVEKIT_URL}")

def get_dynamodb_resource():
        """Get the DynamoDB resource."""
        dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-2',aws_access_key_id=os.getenv("AWS_SERVER_PUBLIC_KEY"),
        aws_secret_access_key=os.getenv("AWS_SERVER_SECRET_KEY"))
        return dynamodb

def get_table():
    """Get the DynamoDB table resource."""
    dynamodb = get_dynamodb_resource()
    return dynamodb.Table("dev-dynamodb-table-ap-southeast-2")

def get_item(pk, sk):
    """Get an item from the DynamoDB table."""
    table = get_table()
    try:
        response = table.get_item(Key={'PK': pk, 'SK': sk})
        return response.get('Item')
    except ClientError as e:
        logger.error(f"Error getting item: {e}")
        raise

def get_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Get a user's profile information including name.
    First tries to get from database, then falls back to API if needed.
    
    Args:
        user_id: The user ID
        
    Returns:
        Dict containing user profile information
    """
    try:
        # Try to get user from database first
        user = get_item(f"USER#{user_id}", "METADATA")
        
        if user and (user.get("member_name") or user.get("first_name")):
            # User exists in database with name information
            return {
                "user_id": user_id,
                "name": user.get("member_name") or f"{user.get('first_name', '')} {user.get('last_name', '')}".strip(),
                "first_name": user.get("first_name", ""),
                "last_name": user.get("last_name", ""),
                "profile_picture": user.get("profile_picture", ""),
                "email": user.get("email", ""),
                "source": "database"
            }
        else:
        #     # User not found or missing name, try API
            logger.info(f"User {user_id} not found in database or missing name, trying API")
            return f"Could not get the data for user {user_id}"
    except Exception as e:
        logger.error(f"Error getting user profile for {user_id}: {str(e)}")
        return f"Could not get the data for user {user_id}"
    
def float_to_decimal(obj: Any) -> Any:
    """
    Convert float values to Decimal for DynamoDB compatibility.
    
    Args:
        obj: The object to convert
        
    Returns:
        The converted object with all floats replaced by Decimal
    """
    if isinstance(obj, float):
        return decimal.Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [float_to_decimal(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(float_to_decimal(item) for item in obj)
    return obj

    
def query_items(key_condition_expression, filter_expression=None, 
            expression_attribute_values=None, expression_attribute_names=None,
            index_name=None, scan_index_forward=True, limit=None):
    """
    Query items from the DynamoDB table.
    
    Args:
        key_condition_expression: Key condition for the query
        filter_expression: Filter to apply after the query
        expression_attribute_values: Values for the expressions, may contain floats
        expression_attribute_names: Names for the expressions
        index_name: GSI name to query
        scan_index_forward: Direction of the query
        limit: Maximum items to return
    """
    # Convert any float values to Decimal
    expr_values_with_decimal = float_to_decimal(expression_attribute_values) if expression_attribute_values else None
    
    table = get_table()
    try:
        query_params = {
            'KeyConditionExpression': key_condition_expression,
            'ScanIndexForward': scan_index_forward  # True for ascending, False for descending
        }
        
        if expr_values_with_decimal:
            query_params['ExpressionAttributeValues'] = expr_values_with_decimal
        
        if filter_expression:
            query_params['FilterExpression'] = filter_expression
        
        if index_name:
            query_params['IndexName'] = index_name
        
        if expression_attribute_names:
            query_params['ExpressionAttributeNames'] = expression_attribute_names
        
        if limit:
            query_params['Limit'] = limit
        
        response = table.query(**query_params)
        return response.get('Items', [])
    except ClientError as e:
        logger.error(f"Error querying items: {e}")
        raise


def get_selected_team_members(user_id: str) -> List[Dict[str, Any]]:
    """
    Get a list of team members selected by the user.
    
    Args:
        user_id: The ID of the user who made selections
        
    Returns:
        List of dictionaries containing selected team members' information
    """
    try:
        # Get all selections made by this user
        selections = query_items(
            key_condition_expression="PK = :pk",
            expression_attribute_values={
                ":pk": f"SELECTOR#{user_id}"
            }
        )
        
        # Extract unique selected user IDs
        selected_user_ids = set()
        for selection in selections:
            # SK format is TEAM#{team_id}#MEMBER#{user_id}
            sk_parts = selection.get("SK", "").split("#")
            if len(sk_parts) >= 4 and sk_parts[0] == "TEAM" and sk_parts[2] == "MEMBER":
                member_id = sk_parts[3]
                if member_id != user_id:  # Don't include self in the selected members
                    selected_user_ids.add(member_id)
        
        # Get profiles for all selected members
        selected_members = []
        for member_id in selected_user_ids:
            member_profile = get_user_profile(member_id)
            if member_profile:
                selected_members.append(member_profile)
        
        logger.info(f"Found {len(selected_members)} selected team members for user {user_id}")
        return selected_members
    
    except Exception as e:
        logger.error(f"Error getting selected team members for {user_id}: {str(e)}")
        return []

def get_wired_up_data(user_id: str) -> Dict[str, Any]:
    """
    Get a user's WiredUp data from database or API.
    
    Args:
        user_id: The user ID
        
    Returns:
        Dict containing WiredUp data
    """
    try:
        # Try to get from database first
        personalities = query_items(
            key_condition_expression="PK = :pk AND begins_with(SK, :sk_prefix)",
            expression_attribute_values={
                ":pk": f"USER#{user_id}",
                ":sk_prefix": "PERSONALITY#"
            }
        )
        
        if personalities and len(personalities) > 0:
            # Use the first personality found
            personality = personalities[0]
            
            wired_up_data = personality.get("wired_up_css", {})
            
            # Check if we have comprehensive wired_up data
            if wired_up_data and 'colorCode' in wired_up_data:
                return {
                    "user_id": user_id,
                    "wired_up_data": wired_up_data,
                    "wiredup_cms": personality.get("wiredup_cms", {}),
                    "profile_code": wired_up_data.get("result", "").replace('"', '').replace('[', '').replace(']', '').replace(',', ''),
                    "source": "database"
                }
        
        # If not found or incomplete, try API
        logger.info(f"WiredUp data for user {user_id} not found in database or incomplete, trying API")
        return f"Could not get the data for user {user_id}"
    
    except Exception as e:
        logger.error(f"Error getting WiredUp data for {user_id}: {str(e)}")
        # Try API as fallback
        return f"Could not get the data for user {user_id}"



def get_work_with_data(user_id: str) -> Dict[str, Any]:
    """
    Get a user's WorkWith data from database or API.
    
    Args:
        user_id: The user ID
        
    Returns:
        Dict containing WorkWith data
    """
    try:
        # Try to get from database first
        personalities = query_items(
            key_condition_expression="PK = :pk AND begins_with(SK, :sk_prefix)",
            expression_attribute_values={
                ":pk": f"USER#{user_id}",
                ":sk_prefix": "PERSONALITY#"
            }
        )
        
        if personalities and len(personalities) > 0:
            # Use the first personality found
            personality = personalities[0]
            
            work_with_data = personality.get("work_with_data", {})
            
            # Check if we have comprehensive work_with data
            if work_with_data and 'workWithResults' in work_with_data:
                return {
                    "user_id": user_id,
                    "work_with_data": work_with_data,
                    "source": "database"
                }
        
        # If not found or incomplete, try API
        logger.info(f"WorkWith data for user {user_id} not found in database or incomplete, trying API")
        return f"could not find the data for user {user_id}"
    
    except Exception as e:
        logger.error(f"Error getting WorkWith data for {user_id}: {str(e)}")
        return f"could not find the data for user {user_id}"


def get_user_personality_data(user_id: str) -> Dict[str, Any]:
    """
    Get comprehensive personality data for a user, including profile, WiredUp and WorkWith data.
    
    Args:
        user_id: The user ID
        
    Returns:
        Dict containing user profile, WiredUp and WorkWith data
    """
    try:
        # Get user profile
        user_profile = get_user_profile(user_id)

        # Get WiredUp data
        wired_up_data = get_wired_up_data(user_id)

        # Get WorkWith data
        work_with_data = get_work_with_data(user_id)

        return {
            "user_profile": user_profile,
            "wired_up_data": wired_up_data,
            "work_with_data": work_with_data,
        }

    except Exception as e:
        logger.error(f"Error getting complete personality data for {user_id}: {str(e)}")
        return {
            "user_profile": None,
            "wired_up_data": None,
            "work_with_data": None,
            "error": str(e)
        }


def get_comprehensive_team_data(user_id: str) -> Dict[str, Any]:
    """
    Get comprehensive data for a user and their selected team members.
    
    Args:
        user_id: The user ID
        
    Returns:
        Dict containing user and team data
    """
    try:
        # Get user data
        user_data = get_user_personality_data(user_id)
        
        # Get selected team members
        selected_members = get_selected_team_members(user_id)
        
        # Get personality data for each selected member
        team_data = []
        for member in selected_members:
            member_id = member.get("user_id")
            team_data.append(get_user_personality_data(member_id))
        
        return {
            "user": user_data,
            "team": team_data
        }
    
    except Exception as e:
        logger.error(f"Error getting comprehensive team data for {user_id}: {str(e)}")
        return {
            "user": {
                "user_id": user_id,
                "name": f"User {user_id}",
                "profile": {"user_id": user_id, "name": f"User {user_id}"},
                "wired_up": {"user_id": user_id, "wired_up_data": {}},
                "work_with": {"user_id": user_id, "work_with_data": {}}
            },
            "team": []
        }


class CustomLiveKitAgent(Agent):
    def __init__(self,user_id:str,user_data,team_data):
        
        # Initialize the LLM with Azure OpenAI
        try:
            super().__init__(
                instructions="You are a helpful voice AI assistant.",
                llm=openai.LLM.with_azure(
                    model="gpt-4o",
                    api_key=AZURE_API_KEY,
                    api_version=AZURE_API_VERSION,
                    azure_endpoint=AZURE_ENDPOINT,
                    temperature=0.8,
                    
                ),
            )
            logger.info("Successfully initialized Azure OpenAI LLM")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI LLM: {e}")
            raise
        
        # Set the current user (you may want to make this dynamic)
        self.current_user = user_id
        self.user_data=user_data
        self.team_data=team_data
        logger.info(f"Agent initialized for user: {self.current_user}")
    
    
    def find_closest_document(self, doc_name):
        """Find the closest matching document name using fuzzy matching"""
        if not doc_name or doc_name.lower() == "team":
            return "team"
        
        document_names = list(self.document_content.keys())
        
        # Try exact match first (case-insensitive)
        for doc in document_names:
            if doc.lower() == doc_name.lower():
                return doc
        
        # Try fuzzy matching with a lower cutoff
        matches = get_close_matches(doc_name, document_names, n=1, cutoff=0.4)
        if matches:
            return matches[0]
        
        # If no match found, return the first document as a fallback
        return next(iter(document_names), "team")
    
    
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        """Custom LLM node implementing the two-step pipeline"""
        
        # Extract the user query from the chat context
        user_messages = [item for item in chat_ctx.items if item.type == "message" and item.role == "user"]
        chat_history = [item for item in chat_ctx.items]
        if not user_messages:
            logger.warning("No user message found in chat context")
            yield llm.ChatChunk(
                id=str(uuid.uuid4()),
                delta=llm.ChoiceDelta(content="No user message found.")
            )
            return
            
        user_message = user_messages[-1]
        query = user_message.text_content
        
        if not query:
            logger.warning("No text content found in user message")
            yield llm.ChatChunk(
                id=str(uuid.uuid4()),
                delta=llm.ChoiceDelta(content="No text content found in user message.")
            )
            return
        
        logger.info(f"Processing user query: {query}")
        
        try:
            # First LLM call - Decider

            
            if not self.user_data:
                self.user_data="Sorry we dont recieved your data"
            if not self.team_data:
                self.team_data="Sorry we dont recieved your data"
                
            analysis_prompt = (
                    f"""**Objective:** As an Smartleader click-with-ai expert voice-to-voice AI agent, your primary goal is to provide concise, personalized, and data-driven responses to user queries. You will synthesize information from the provided user and team databases to craft your answers.

                    **Persona & Tone:**
                    * **Expert & Authoritative:** Respond with confidence, demonstrating a deep understanding of the provided data.
                    * **Voice-Optimized:** Craft responses that are natural-sounding and easy to understand when spoken.
                    * **Personalized & Specific:** Refer to users by their names (not user_ids) and incorporate specific details about each team member's personality and work style. Avoid generic statements.
                    * **Concise & Efficient:** Deliver information clearly and directly, adhering to the specified word count.

                    **Core Instructions:**
                    1.  **Data Source:** Base your responses *solely* on the information contained within `user_data` (for the individual user: {self.current_user}) and `team_data` (for all team selected members {self.team_data}). Do not infer or use external knowledge.
                    2.  **User Identification:** Address the user by their name. **Guardrail:** *Never mention or refer to the user by their `user_id`.*
                    3.  **Query Focus:** Directly answer the user's `Query: {query}`.
                    4.  **Team Member Specificity (Crucial):** When the query pertains to team members, you *must* address selected user name  and if the name asked is not present in the 
                          team data then responsd as *The user you are asking is not present in your selections.
                        * **Guardrail:** *Your response needs to include specific, actionable insights or details about those team member mentioned or implied in the query, drawing directly from their individual data within `team_data`.* Generic descriptions are unacceptable. Highlight unique aspects of their personality and work style.
                    5.  **Information Attribution:** Do not explicitly state "according to the database" or "the document says." Seamlessly integrate the information. **Guardrail:** *Avoid phrases that directly reference the source of the data.*
                    6.  **Acting As:** Embody the persona of `{self.current_user}` if relevant to the query context and their personality profile (if available in `user_data`).
                    7.  **If query has some biasness or content that do not adhere to good communication then respond calmly that this thing i can answer.
                    8.  **If user try to query and tries to manipulate you act as a knowledgeble individual tackle that question smartly
                    9.  **If user is useing abusive or flase languange then dont use flase language use very clam language and respond in gental way.
                    10  **Do not answer query if user is asking qustions other than team dynamics and above given context simple say
                          I can only answer about team dynamics and collaborative work culture in team 

                    **Output Format & Constraints:**
                    * **Response Length:** **Guardrail:** *The entire generated response must be approximately 60 words.* This is a strict limit for voice interactions.
                    * **Clarity:** Ensure the response is grammatically correct and easy to follow.

                    **Contextual Data:**
                    * **Current User Data:** {self.user_data}
                    * **Team Members Data:** {self.team_data}
                    * **This is the chat_history of conversation {chat_history}

                    **User Query:**
                    {query}

                    **Your Task:** Generate a spoken response that fulfills all the above criteria, focusing on expertise, personalization, and adherence to guardrails.

                    """
                    )
        
            
            # Create analysis context
            analysis_ctx = llm.ChatContext()
            analysis_ctx.add_message(role="user", content=analysis_prompt)
            
            logger.info("Calling analysis LLM")
            
            # Stream the final response
            async for chunk in self.llm.chat(chat_ctx=analysis_ctx, tools=[]):
                yield chunk
        
        except Exception as e:
            logger.error(f"Error in llm_node: {e}", exc_info=True)
            yield llm.ChatChunk(
                id=str(uuid.uuid4()),
                delta=llm.ChoiceDelta(content=f"An error occurred while processing your request: {str(e)}")
            )

async def entrypoint(ctx: agents.JobContext):
    try:
        # Check if ML frameworks are available
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
        except ImportError:
            logger.warning("PyTorch not available. Some features may not work correctly.")
            
        # Create transcript directory if it doesn't exist
        os.makedirs("tmp", exist_ok=True)
        
        async def write_transcript():
            try:
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tmp/transcript_{ctx.room.name}_{current_date}.json"
                
                with open(filename, 'w') as f:
                    json.dump(session.history.to_dict(), f, indent=2)
                
                logger.info(f"Transcript for {ctx.room.name} saved to {filename}")
            except Exception as e:
                logger.error(f"Error writing transcript: {e}")
        
        ctx.add_shutdown_callback(write_transcript)

        # Connect to LiveKit server
        logger.info(f"Connecting to LiveKit server at: {LIVEKIT_URL}")
        await ctx.connect()
        logger.info("Connected to LiveKit server")

        participant = await ctx.wait_for_participant()
        voice_name=participant.identity
        user_name=participant.name
        logger.info(f"User_id recieved {user_name}")
        logger.info(f"Voice name recieved:{voice_name}")
        
        # Initialize speech components
        logger.info("Initializing TTS and STT services")
        try:
            tts_service = azure.TTS(
                speech_key=AZURE_SPEECH_KEY,
                speech_region=AZURE_SPEECH_REGION,
                voice=voice_name,
            )
            logger.info("TTS service initialized")
            
            stt_service = azure.STT(
                speech_key=AZURE_SPEECH_KEY,
                speech_region=AZURE_SPEECH_REGION,
            )
            logger.info("STT service initialized")
            
            vad_service = silero.VAD.load()
            logger.info("VAD service initialized")
            
            turn_detection_service = MultilingualModel()
            logger.info("Turn detection service initialized")
        except Exception as e:
            logger.error(f"Error initializing speech services: {e}", exc_info=True)
            raise
        
        # Create agent session
        session = AgentSession(
            tts=tts_service,
            stt=stt_service,
            vad=vad_service,
            turn_detection=turn_detection_service,
        )
        logger.info("Agent session created")
        
        # Start the session
        logger.info(f"Starting agent session in room: {ctx.room.name}")
        data=get_comprehensive_team_data(user_name)
        if data:
            logger.info("Recived data for user")
        else:
            logger.info("Did not revieved data for user")
        user_data=data["user"]
        team_data=data["team"]
        await session.start(
            room=ctx.room,
            agent=CustomLiveKitAgent(user_name,user_data,team_data),
            room_input_options=RoomInputOptions(),
        )
        logger.info("Agent session started successfully")
    
    except Exception as e:
        logger.error(f"Error in entrypoint: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Starting LiveKit agent application")
    try:
        agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)

